from typing import Dict, Any, List, Optional, Union, Tuple
from collections.abc import Iterator
import random
import json
import os
import time
import torch
import hashlib
import base64
import traceback
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import IterableDataset
from fastparquet import ParquetFile
import pandas as pd
from muse.training.parallel import get_data_parallel_rank, get_data_parallel_world_size

PARQUET_CACHE_DIR = os.environ.get("PARQUET_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "muse", "parquet"))

def is_hdfs(path: str) -> bool:
  """Check if a path is a HDFS path"""
  return path.startswith('viewfs://') or path.startswith('hdfs://')

def get_worker_info():
  worker = 0
  num_workers = 1
  try:
    import torch.utils.data
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      worker = worker_info.id
      num_workers = worker_info.num_workers
  except ModuleNotFoundError:
    pass
  return worker, num_workers

def is_image_exist(image_path: str) -> bool:
  return bool(image_path) and os.path.exists(image_path) \
    and os.path.getsize(image_path) > 0

def load_image(image: str) -> Optional[Image.Image]:
  try:
    if not isinstance(image, str):
      return None
    if not is_image_exist(image):
      image_bytes = base64.b64decode(image)
      image = Image.open(BytesIO(image_bytes))
    else:
      image = Image.open(image)
    return image
  except (ValueError, OSError, IOError) as e:
    print(f"Error loading image: {e}")
    return None

def rename(src: str, dst: str):
  if is_hdfs(src) and is_hdfs(dst):
    cmd = f'hadoop fs -mv {src} {dst}'
    os.system(cmd)
  else:
    os.system(f"mv {src} {dst}")

def calculate_text_hash(text):
  hash_object = hashlib.sha256()
  hash_object.update(text.encode('utf-8'))
  return hash_object.hexdigest()

def load_parquet(path: str) -> ParquetFile:
  """Load a parquet file, with fallback to local cache if HDFS read fails."""
  rank = get_data_parallel_rank()
  worker, _ = get_worker_info()
  retry: int = 10
  max_cache_files: int = 10

  cache_dir = os.path.join(PARQUET_CACHE_DIR, f'{rank}_{worker}')
  os.makedirs(cache_dir, exist_ok=True)
  filename = os.path.basename(path)

  cache_fn = os.path.join(
    cache_dir, str(calculate_text_hash(path)) + '_' + filename)

  def clean_cache_if_needed():
    files = [
      os.path.join(cache_dir, f) for f in os.listdir(cache_dir) \
        if os.path.isfile(os.path.join(cache_dir, f))]
    if len(files) > max_cache_files:
      files.sort(key=os.path.getctime)
      file_to_remove = max(len(files) - max_cache_files, 0)
      for fn in files[:file_to_remove]:
        print(f"Removing old cached file: {fn}")
        os.remove(fn)

  for r in range(retry):
    clean_cache_if_needed()
    try:
      if os.path.exists(path):
        return ParquetFile(path)
      elif os.path.exists(cache_fn):
        return ParquetFile(cache_fn)
      else:
        raise FileNotFoundError("File not found")
    except (OSError, IOError, FileNotFoundError, RuntimeError) as e:
      try:
        cmd = f'hadoop fs -get {path} {cache_fn}'
        if os.system(cmd) != 0:
          raise RuntimeError("HDFS get command failed")
        return ParquetFile(cache_fn)
      except (OSError, IOError, RuntimeError) as e2:
        time.sleep(2 + np.random.randint(0, 5))
        if r == retry - 1:
          raise IOError('parquet', path, 
            f"Failed to load from both original path and cache. "
            f"Original error: {e}\nCache error: {e2}")
    print(f"Retrying for path={path} / {cache_fn}")

class Reader(Iterator):
  def __init__(self,
               sources: Union[List[str], str]):
    if isinstance(sources, str):
      sources = [sources]
    self.sources = sources
  
  def __iter__(self):
    return self
  
  def __next__(self):
    raise NotImplementedError("Subclass must implement this method")

class ParquetReader(Reader):
    """
    IterableDataset for parquet files, consuming files in order.
    """
    def __init__(self, sources: Union[List[str], str]):
      super().__init__(sources)

    def _parser(self,
                row: Dict[str, Any],
                filename: str,
                index: int,
                size: int) -> Dict[str, Any]:
      row["__file__"] = filename
      row["__index__"] = index
      row["__total__"] = size
      return row

    def __iter__(self,):
      rank = get_data_parallel_rank()
      worker_id, _  = get_worker_info()
      for fn in tqdm(self.sources):
        try:
          parquet_file = load_parquet(fn)
        except Exception as e:
          print(f"open parquet fail {fn=}, error_msg={traceback.format_exc()}")
          continue
        df = parquet_file.to_pandas()
        for idx, row in tqdm(
            df.iterrows(), total=len(df),
            desc=f"[rank={rank}, worker={worker_id}] {fn}"):
          try:
            sample = self._parser(row.to_dict(), fn, idx, len(df))
            if sample is not None:
              yield sample
          except Exception as e:
            print(f"Error processing row {idx} in {fn}: {str(e)}")
            continue


class ShuffledParquetReaderV2(Reader):
    """
    IterableDataset for parquet files, consuming files in order.
    """
    def __init__(self, sources: Union[List[str], str], local_shuffle_buffer_size=4096*4, local_shuffle_random_fetch=0.01):
      super().__init__(sources)
      from .local_shuffle_buffer import LocalShuffleBuffer
      self.local_shuffle_buffer = LocalShuffleBuffer(
        local_shuffle_buffer_size, local_shuffle_random_fetch)


    def _parser(self,
                row: Dict[str, Any],
                filename: str,
                index: int,
                size: int) -> Dict[str, Any]:
      row["__file__"] = filename
      row["__index__"] = index
      row["__total__"] = size
      return row

    def __iter__(self,):
      rank = get_data_parallel_rank()
      worker_id, _  = get_worker_info()
      for fn_index, fn in tqdm(enumerate(self.sources)):
        try:
          parquet_file = load_parquet(fn)
        except Exception as e:
          print(f"open parquet fail {fn=}, error_msg={traceback.format_exc()}")
          continue
        df = parquet_file.to_pandas()

        # todo: pass a true epoch_idx here
        df = self.local_shuffle_buffer.preprocess_df(df, epoch_idx=0, fn_index=fn_index, fn=fn)
        
        for idx, row in tqdm(
            df.iterrows(), total=len(df),
            desc=f"[rank={rank}, worker={worker_id}] {fn}"):
          try:
            sample = self._parser(row.to_dict(), fn, idx, len(df))
            if sample is not None:
              if self.local_shuffle_buffer.add(sample, fn, log_info=f"rank{rank}-{fn}", index=idx): continue
              sample = self.local_shuffle_buffer.get()
              yield sample

          except Exception as e:
            print(f"Error processing row {idx} in {fn}: {str(e)}")
            continue



class ShuffledParquetReader(ParquetReader):
    """
    带局部窗口打乱的 Parquet 读取器。
    通过维护一个滑动的文件窗口并重新对窗口内的所有行进行打乱，来增加数据的随机性。
    """
    def __init__(self, sources: Union[List[str], str], window_size: int = 5):
        super().__init__(sources)
        self.window_size = window_size

    def __iter__(self):
        rank = get_data_parallel_rank()
        worker_id, _ = get_worker_info()
        
        if not self.sources:
            return

        file_list = list(self.sources)
        # 初始加载窗口
        current_window_dfs = []
        file_idx = 0
        
        # 记录窗口中每个文件贡献的行数，用于触发“滑出”逻辑
        rows_per_file = []

        def load_next_df():
            nonlocal file_idx
            while file_idx < len(file_list):
                fn = file_list[file_idx]
                file_idx += 1
                try:
                    pf = load_parquet(fn)
                    df = pf.to_pandas()
                    if len(df) > 0:
                        df["__file_origin__"] = fn # 记录来源用于解析
                        return df
                except Exception as e:
                    print(f"Error loading {fn}: {e}")
            return None

        # 1. 填充初始窗口
        for _ in range(min(self.window_size, len(file_list))):
            df = load_next_df()
            if df is not None:
                current_window_dfs.append(df)
                rows_per_file.append(len(df))

        if not current_window_dfs:
            return

        # 合并并执行第一次打乱
        combined_df = pd.concat(current_window_dfs, ignore_index=True)
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        rows_processed_since_update = 0

        # 2. 开始迭代
        while True:
            for idx, row in combined_df.iterrows():
                # 解析并产出
                sample = self._parser(row.to_dict(), row["__file_origin__"], idx, len(combined_df))
                if sample is not None:
                    yield sample
                
                rows_processed_since_update += 1

                # 3. 检查是否处理完了一个文件的等量数据，需要滑动窗口
                # 当处理行数达到窗口中“最老”文件的行数时，替换数据
                if rows_processed_since_update >= rows_per_file[0]:
                    # 如果还有后续文件，则补充
                    next_df = load_next_df()
                    
                    # 移除旧文件行（简单处理：重新构建 df 防止内存碎片或偏移混乱）
                    # 这里的逻辑是：去掉前 rows_per_file[0] 行，加上新文件，再打乱
                    if next_df is not None:
                        # 窗口滑动：去掉已消耗的部分，加入新数据
                        remaining_df = combined_df.iloc[idx + 1:]
                        combined_df = pd.concat([remaining_df, next_df], ignore_index=True)
                        combined_df = combined_df.sample(frac=1).reset_index(drop=True)
                        
                        rows_per_file.pop(0)
                        rows_per_file.append(len(next_df))
                        rows_processed_since_update = 0
                        break # 跳出当前的 iterrows，从新的 combined_df 开始
                    else:
                        # 没有新文件了，继续消耗完当前窗口剩下的
                        rows_per_file.pop(0)
                        rows_processed_since_update = 0
                        # 如果所有文件都处理完且 rows_per_file 为空，则退出
                        if not rows_per_file:
                            return
            else:
                # 正常消耗完整个 combined_df 且没有 break 的情况（即最后几个文件）
                return


class DistributedDataset(IterableDataset):
  def __init__(self,
               sources: Union[List[str], str],
               rank: int = 0,
               world_size: int = 1,
               num_workers: int=8,
               seed: int=1024,
               num_epochs: int=1,
               shard_by: str = "auto",
               reader: str = "parquet",
               shuffle_buffer_size: int = 0,
               enable_checkpointing: bool = False,
               shuffle_window: int = 5,
               packing: bool = False,
               padding: bool = False,
               balancing: bool = False,
               max_length: int = 0,
               **kwargs):
    """
    Distributed dataset supporting three sharding modes:
    - shard_by="auto": Auto-select (default), uses 'files' if num_files >= total_workers, else 'samples'
    - shard_by="files": Shard by files, each worker processes different files (high I/O efficiency, suitable for sufficient files)
    - shard_by="samples": Shard by samples, all workers read same files but process different samples (better load balancing, lower I/O efficiency)
    
    Recommended usage:
    - Local/single-node multi-GPU training: prefer "files" (avoid I/O contention)
    - Distributed training + HDFS storage: prefer "files" (save network bandwidth)
    - Very few files but many samples: use "samples"
    - Uncertain: use "auto" (default)
    """
    self.rank = rank
    self.world_size = world_size
    self.num_workers = num_workers
    self.seed = seed
    self.shuffle_buffer_size = shuffle_buffer_size
    self.enable_checkpointing = enable_checkpointing
    self.packing = packing
    self.max_length = max_length
    self.shuffle_window = shuffle_window
    if self.packing:
      assert self.max_length > 0, "max_length must be set when packing is enabled"
    self.padding = padding
    self.balancing = balancing
    assert shard_by in ["auto", "files", "samples"], \
      f"shard_by must be 'auto', 'files' or 'samples', got {shard_by}"

    self.rng = random.Random(seed)
    self.num_epochs = num_epochs
    self.sources = sources
    self.shard_by = shard_by
    self.reader = reader
    self.kwargs = kwargs
    self.shuffle_window = shuffle_window
    # Initialize attributes
    self._files = []
    self._actual_shard_by = shard_by

    self.dataset = self._build()

  def _load_file_list(self) -> List[str]:
    """Load file list"""
    files = []
    if isinstance(self.sources, list):
        files = self.sources
    elif self.sources.endswith(".json"):
      with open(self.sources, "r") as fp:
        files = json.loads(fp.read())
        files = sorted([
          fn for fn in files if fn.endswith(".parquet")])
    else:
      # TODO: support hdfs
      folder = Path(self.sources)
      files = list(map(str, folder.rglob("*.parquet")))
    
    return files

  # def _get_reader_class(self):
  #   if self.reader == "parquet":
  #     return ParquetReader
  #   else:
  #     raise ValueError(f"Unsupported reader: {self.reader}")


  def _get_reader_class(self):
    # 根据配置返回对应的 Reader
    if self.reader == "parquet":
      if self.shuffle_window > 1:
        # 返回一个构造函数，支持传入 window_size
        # return lambda sources: ShuffledParquetReaderV2(sources, local_shuffle_buffer_size=max(1024 * self.shuffle_window, 4096))
        # return lambda sources: ShuffledParquetReader(sources, window_size=min(self.shuffle_window, 5))
        
        print(f"Shuffle window: {self.shuffle_window}")
        # 返回一个构造函数，支持传入 window_size
        return lambda sources: ShuffledParquetReader(sources, window_size=self.shuffle_window)
      return ParquetReader
    else:
      raise ValueError(f"Unsupported reader: {self.reader}")

  def _build(self):
    """Build dataset based on shard_by mode"""
    files = self._load_file_list()
    assert len(files) > 0, f"No file found for rank{self.rank}"

    # Shuffle file list with fixed seed 
    # (all ranks and workers use same seed to ensure consistent order)
    files_rng = random.Random(self.rng.getstate()[1][0])  # Use initial seed
    files_rng.shuffle(files)

    # Calculate total parallelism
    total_workers = self.world_size * self.num_workers
    
    # Determine actual sharding mode
    actual_shard_by = self.shard_by
    
    if self.shard_by == "auto":
      # Auto-select mode
      if len(files) >= total_workers:
        actual_shard_by = "files"
        if self.rank == 0:
          print(f"📂 Auto-selected 'files' mode: {len(files)} files >= {total_workers} workers")
      else:
        actual_shard_by = "samples"
        if self.rank == 0:
          print(f"📊 Auto-selected 'samples' mode: {len(files)} files < {total_workers} workers")
    elif self.shard_by == "files" and len(files) < total_workers:
      # Force files mode but switch to samples when files are insufficient
      actual_shard_by = "samples"
      if self.rank == 0:
        print(
          f"⚠️  Warning: File count ({len(files)}) < total workers ({total_workers}), "
          f"auto-switching from 'files' to 'samples' mode")

    if actual_shard_by == "files":
      # Shard by files: each (rank, worker) combination processes different files
      # Calculate current worker's global index
      # Note: actual worker_id can only be obtained in __iter__, store all files here first
      # Actual sharding will be done in __iter__
      
      print(
        f"DistributedDataset [shard_by=files] "
        f"rank={self.rank}/{self.world_size}, num_workers={self.num_workers}, "
        f"total_files={len(files)}, total_parallel_workers={total_workers}")
      
      # Store all files, distribute by worker in __iter__
      self._files = files
      self._actual_shard_by = "files"
      dataset = None  # Lazy creation
    else:
      # Shard by samples: all ranks/workers process same files
      print(
        f"DistributedDataset [shard_by=samples] "
        f"rank={self.rank}/{self.world_size}, num_workers={self.num_workers}, "
        f"total_files={len(files)}")

      self._actual_shard_by = "samples"
      dataset = self._get_reader_class()(sources=files)
    
    return dataset

  def process(self,
              sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    raise NotImplementedError("Subclass must implement this method")

  def _get_reader_iter(self):
    worker_id, num_workers = get_worker_info()
    
    if self._actual_shard_by == "files":
      # File sharding mode: each (rank, worker) combination processes different files
      # Calculate current worker's global index
      global_worker_id = self.rank * self.num_workers + worker_id
      total_workers = self.world_size * self.num_workers
      
      # Distribute files by global worker index
      files_for_this_worker = self._files[global_worker_id::total_workers]

      print(
        f"  Worker[rank={self.rank}, worker={worker_id}, global={global_worker_id}] "
        f"processing {len(files_for_this_worker)} files")
      
      # Create dataset for current worker
      reader = self._get_reader_class()(files_for_this_worker)
      
      # Process all samples sequentially (files already distributed)
      for sample in reader:
        yield sample
    else:
      # Sample sharding mode: need to consider both rank and worker sharding
      for idx, sample in enumerate(self.dataset):
        if idx % self.world_size == self.rank and \
                idx % num_workers == worker_id:
          yield sample

  def process(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    raise NotImplementedError("Subclass must implement this method")

  def pack_sample(self,
                  inputs: Dict[str, torch.Tensor],
                  new_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    raise NotImplementedError("Subclass must implement this method")
  
  def get_sample_length(self, sample: Dict[str, torch.Tensor]) -> int:
    raise NotImplementedError("Subclass must implement this method")

  def __iter__(self):
    """Iterate through the dataset, processing samples and handling epochs."""
    import signal
    import traceback
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Timeout handler for stuck samples
    def timeout_handler(signum, frame):
      raise TimeoutError("Sample processing timeout (120 secs)")

    # Error tracking per data source
    source_sample_cnt = {}
    source_error_cnt = {}

    buffer = []
    source_list = []  # Track data sources for each sample in buffer
    current_length = 0
    for _ in range(self.num_epochs):
      for sample in self._get_reader_iter():
        # Get source name for error tracking
        source_name = "unknown"
        sample_key = ""
        sample_url = ""
        try:
          if isinstance(sample, dict):
            source_name = sample.get("source", sample.get("json", {}).get("source", "unknown"))
            sample_key = sample.get("__key__", sample.get("uuid", ""))
            sample_url = sample.get("__url__", sample.get("__file__", ""))
        except:
          pass

        source_sample_cnt.setdefault(source_name, 0)
        source_sample_cnt[source_name] += 1

        try:
          # Set timeout for processing (Unix-only, gracefully skip on Windows)
          signal.signal(signal.SIGALRM, timeout_handler)
          signal.alarm(120)  # 增加到5分钟超时
          
          new_inputs = self.process(sample)
          
          # Clear timeout
          try:
            signal.alarm(0)
          except (AttributeError, ValueError):
            pass
          
        except StopIteration:
          # Clear timeout and re-raise
          try:
            signal.alarm(0)
          except (AttributeError, ValueError):
            pass
          raise

        except Exception as e:
          # Clear timeout
          try:
            signal.alarm(0)
          except (AttributeError, ValueError):
            pass
          
          # Track errors
          source_error_cnt.setdefault(source_name, 0)
          source_error_cnt[source_name] += 1
          error_ratio = source_error_cnt[source_name] * 1.0 / source_sample_cnt[source_name]
          
          # Log error occasionally to avoid flooding logs
          if np.random.rand() < 1:  # 1% chance to log
            logger.error(
              f"DistributedDataset process sample error. "
              f"{source_name=}, {error_ratio=:.4f}, {sample_key=}, {sample_url=}\n"
              f"errmsg={traceback.format_exc()}"
              f"sample={sample}"
            )
          continue  # Skip bad sample and continue
        
        if not new_inputs:
          continue
        if self.packing:
          new_sample_length = self.get_sample_length(new_inputs)
          if current_length + new_sample_length > self.max_length:
            packed_inputs = self.pack_sample(buffer)
            # Add data_source info for monitoring
            packed_inputs["data_source"] = source_list
            yield packed_inputs
            buffer = []
            source_list = []
            current_length = 0
          buffer.append(new_inputs)
          source_list.append(source_name)
          current_length += new_sample_length
        else:
          # For non-packing mode, add source info directly
          new_inputs["data_source"] = [source_name]
          yield new_inputs
    if self.packing and len(buffer) > 0:
      packed_inputs = self.pack_sample(buffer)
      packed_inputs["data_source"] = source_list
      yield packed_inputs