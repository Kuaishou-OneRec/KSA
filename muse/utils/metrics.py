"""
Metrics Management System for Training Monitoring with Distributed Support.

This module provides a flexible metrics tracking system for distributed deep 
learning training workflows with computation graph support. Key features:

- Series with configurable fill values and initial values
- Distributed reduction across ranks (mean/sum)
- **Unified None handling: all missing values use None**
- **Automatic None exclusion in all computations (avg, sum, cumsum, etc.)**
- Step-based synchronization similar to TensorFlow's session.run()
- Lazy evaluation with automatic cache invalidation
- Sliding window operations
- Multi-level dependency support (nested computations)

Missing Value Handling:
    All missing values are None. Computations automatically exclude None:
    - avg() and sum() skip None values
    - Distributed reduction excludes None values from aggregation
    - Arithmetic operations propagate None (None + x = None)
    - cumsum() skips None values and continues accumulation

Process Group Support:
    - Metrics maintains a single process_group
    - All series share this process_group for distributed reduction

Classes:
    Scalar: Scalar value wrapper for numeric values
    BaseSeries: Abstract base class for all series types
    Series: Primary time series container with distributed reduction support
    DerivedSeries: Computed series that lazily evaluates from source
    Metrics: Main class for managing metrics with shared index and distributed sync

Example:
    >>> # Create metrics with distributed reduction
    >>> metrics = Metrics()
    >>> metrics.new("loss", dtype="float", reduce="mean", initial_value=0.0)
    >>> metrics.initialize()
    >>> 
    >>> # Each rank appends local loss
    >>> metrics.loss.append(local_loss_value)
    >>> 
    >>> # step() performs reduction across ranks
    >>> metrics.step()
    >>> print(metrics.loss[-1])  # Mean across all ranks
    >>> 
    >>> # Create derived series with sliding window
    >>> loss_avg = metrics.loss.avg(window=20)
    >>> print(f"Step 50: {loss_avg[50]}")
"""
import time
from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, List, Dict, Any
import torch
import torch.distributed as dist


class LoggerBackend(ABC):
    """
    Abstract base class for logger backends.
    
    Logger backends handle writing metrics to different outputs (TensorBoard,
    Wandb, CSV files, or stdout). Each backend implements the write() method
    to persist metrics in their specific format.
    """
    
    @abstractmethod
    def write(self, step: int, values: Dict[str, Dict[str, Any]]):
        """
        Write metrics to the backend.
        
        Args:
            step: Current step/index value
            values: Nested dict of {group: {metric_name: value}}
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close and cleanup backend resources."""
        pass


class TensorBoardBackend(LoggerBackend):
    """TensorBoard logging backend using torch.utils.tensorboard."""
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard backend.
        
        Args:
            log_dir: Directory to write TensorBoard logs
        """
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
    
    def write(self, step: int, values: Dict[str, Dict[str, Any]]):
        """Write metrics to TensorBoard."""
        for group, metrics in values.items():
            for name, value in metrics.items():
                tag = f"{group}/{name}" if group else name
                # Extract scalar value if it's a Scalar object
                if hasattr(value, 'value'):
                    value = value.value
                self.writer.add_scalar(tag, value, step)
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


class WandbBackend(LoggerBackend):
    """Weights & Biases logging backend."""
    
    def __init__(self, project: str, run_name: Optional[str] = None, **kwargs):
        """
        Initialize Wandb backend.
        
        Args:
            project: Wandb project name
            run_name: Optional run name
            **kwargs: Additional arguments passed to wandb.init()
        """
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project, name=run_name, **kwargs)
        except ImportError:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
    
    def write(self, step: int, values: Dict[str, Dict[str, Any]]):
        """Write metrics to Wandb."""
        flat_dict = {}
        for group, metrics in values.items():
            for name, value in metrics.items():
                key = f"{group}/{name}" if group else name
                # Extract scalar value if it's a Scalar object
                if hasattr(value, 'value'):
                    value = value.value
                flat_dict[key] = value
        
        self.wandb.log(flat_dict, step=step)
    
    def close(self):
        """Finish Wandb run."""
        if self.run:
            self.run.finish()


class CSVBackend(LoggerBackend):
    """CSV file logging backend."""
    
    def __init__(self, csv_path: str):
        """
        Initialize CSV backend.
        
        Args:
            csv_path: Path to CSV file
        """
        self.csv_path = csv_path
        self.file = open(csv_path, 'w')
        self.headers_written = False
        self.headers = []
    
    def write(self, step: int, values: Dict[str, Dict[str, Any]]):
        """Write metrics to CSV."""
        import csv
        
        # Flatten values
        flat_dict = {'step': step}
        for group, metrics in values.items():
            for name, value in metrics.items():
                key = f"{group}/{name}" if group else name
                # Extract scalar value if it's a Scalar object
                if hasattr(value, 'value'):
                    value = value.value
                flat_dict[key] = value
        
        # Write headers if first time
        if not self.headers_written:
            self.headers = list(flat_dict.keys())
            writer = csv.DictWriter(self.file, fieldnames=self.headers)
            writer.writeheader()
            self.headers_written = True
        
        # Write row
        writer = csv.DictWriter(self.file, fieldnames=self.headers)
        writer.writerow(flat_dict)
        self.file.flush()
    
    def close(self):
        """Close CSV file."""
        self.file.close()


class StdoutBackend(LoggerBackend):
    """Standard output logging backend."""
    
    def __init__(self, prefix: str = "[Metrics]"):
        """
        Initialize Stdout backend.
        
        Args:
            prefix: Prefix string for log lines
        """
        self.prefix = prefix

    def _format_value(self, value: Any) -> str:
        if isinstance(value, float):
            abs_value = abs(value)
            if abs_value != 0.0 and abs_value < 1e-3:
                return f"{value:.3e}"
            return f"{value:.3f}"
        return str(value)
    
    def write(self, step: int, values: Dict[str, Dict[str, Any]]):
        """Write metrics to stdout."""
        metrics_str = []
        for group, metrics in values.items():
            for name, value in metrics.items():
                key = f"{group}/{name}" if group else name
                # Extract scalar value if it's a Scalar object
                if hasattr(value, 'value'):
                    value = value.value
                metrics_str.append(f"{key}={self._format_value(value)}")
        
        print(f"{self.prefix} Step {step}: {', '.join(metrics_str)}")
    
    def close(self):
        """No-op for stdout."""
        pass


class Logger:
    """
    Logger for tracking and writing metrics to multiple backends.
    
    Logger manages a set of tracked series and writes their latest values to
    configured backends (TensorBoard, Wandb, CSV, or stdout) when write() is called.
    
    Example:
        >>> logger = Logger("training", [
        ...     TensorBoardBackend("runs/exp1"),
        ...     StdoutBackend()
        ... ])
        >>> logger.track(metrics.loss, name="loss", group="training")
        >>> logger.write()  # Writes to all backends
    """
    
    def __init__(self, name: str, backends: List[LoggerBackend]):
        """
        Initialize Logger with name and backends.
        
        Args:
            name: Logger name (for identification)
            backends: List of backend instances
        """
        self.name = name
        self.backends = backends
        self._tracked_series: Dict[str, tuple] = {}  # name -> (series, group)
    
    def track(self, series: "BaseSeries", name: str, group: str = ""):
        """
        Track a series for logging.
        
        Args:
            series: Series or DerivedSeries to track
            name: Metric name for logging
            group: Optional group name for organizing metrics
        """
        self._tracked_series[name] = (series, group)
    
    def write(self, step: int):
        """
        Write latest values of all tracked series to backends.
        
        Args:
            step: Current step value
        """
        # Collect values from tracked series
        values: Dict[str, Dict[str, Any]] = {}
        
        for name, (series, group) in self._tracked_series.items():
            if len(series) > 0:
                # Get latest value
                latest_value = series[-1]
                
                # Skip None values (missing data points from slicing operations)
                if latest_value is None:
                    continue
                
                # Convert tensor to scalar if needed (safety check)
                if isinstance(latest_value, torch.Tensor):
                    latest_value = latest_value.item()
                
                # Organize by group
                if group not in values:
                    values[group] = {}
                values[group][name] = latest_value
        
        # Write to all backends (only if there are values to write)
        if values:
            for backend in self.backends:
                backend.write(step, values)
    
    def close(self):
        """Close all backends."""
        for backend in self.backends:
            backend.close()


class LoggerProxy:
    """
    Proxy for broadcasting operations to multiple loggers.
    
    LoggerProxy allows metrics.logger.track() and metrics.logger.write()
    to broadcast to all registered loggers in the Metrics instance.
    """
    
    def __init__(self, loggers: List[Logger]):
        """
        Initialize LoggerProxy.
        
        Args:
            loggers: List of Logger instances to broadcast to
        """
        self._loggers = loggers
    
    def track(self, series: "BaseSeries", name: str, group: str = ""):
        """
        Track a series in all loggers.
        
        Args:
            series: Series or DerivedSeries to track
            name: Metric name for logging
            group: Optional group name for organizing metrics
        """
        for logger in self._loggers:
            logger.track(series, name, group)
    
    def write(self):
        """Write to all loggers using their parent Metrics' current index."""
        # Get the step from the first logger's tracked series' metrics
        # This assumes all loggers share the same Metrics instance
        if self._loggers and len(self._loggers) > 0:
            # Need to get step from somewhere - we'll pass it as parameter
            # For now, raise error if called without step
            raise RuntimeError("LoggerProxy.write() needs access to Metrics step. Use metrics.write_logs() instead.")
    
    def close(self):
        """Close all loggers."""
        for logger in self._loggers:
            logger.close()


class Scalar:
    """
    Scalar value wrapper for numeric values.
    
    Scalar wraps a single numeric value (int, float, or timestamp) and provides
    type conversion and access methods.
    
    Args:
        value (int or float): The numeric value to wrap
        dtype (Literal["int", "float", "timestamp"]): Data type specification.
            "int" values are converted to int, "float" and "timestamp" to float.
    
    Attributes:
        dtype (str): The data type of the scalar
        
    Example:
        >>> scalar = Scalar(3.14, dtype="float")
        >>> print(scalar.value)
        3.14
        >>> print(float(scalar))
        3.14
    """
    
    def __init__(self, value: Union[int, float], dtype: Literal["int", "float", "timestamp"]):
        """
        Initialize a Scalar with automatic type conversion.
        
        Args:
            value (int or float): The numeric value to wrap
            dtype (Literal["int", "float", "timestamp"]): Data type specification
        """
        self._value = value
        self.dtype = dtype
        # Convert value based on dtype
        if dtype == "int":
            self._value = int(value)
        elif dtype == "float" or dtype == "timestamp":
            self._value = float(value)
    
    @property
    def value(self) -> Union[int, float]:
        """
        Get the raw numeric value.
        
        Returns:
            int or float: The underlying value, guaranteed to be int for dtype="int",
                float for dtype="float" or dtype="timestamp"
        """
        if self.dtype == "int":
            return int(self._value)
        else:
            return float(self._value)
    
    def __float__(self) -> float:
        """Convert scalar to float."""
        return float(self._value)
    
    def __int__(self) -> int:
        """Convert scalar to int."""
        return int(self._value)
    
    def __repr__(self) -> str:
        """Return string representation of the Scalar."""
        return f"Scalar(value={self._value}, dtype={self.dtype})"


class BaseSeries(ABC):
    """
    Abstract base class defining the interface for all Series types.
    
    BaseSeries provides a common interface for both Series (which stores actual data)
    and DerivedSeries (which computes data from sources). All series types support:
    
    - Indexing and slicing via __getitem__
    - Length queries via __len__
    - Iteration via __iter__
    - dtype attribute for type information
    - Statistical operations (avg, sum, cumsum) that return DerivedSeries
    
    This abstraction enables treating both stored and computed series uniformly,
    supporting multi-level computation graphs.
    
    Attributes:
        dtype (str): Data type of values in the series ("int", "float", or "timestamp")
    """
    
    @property
    @abstractmethod
    def dtype(self) -> str:
        """Return the data type of values in this series."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the series."""
        pass
    
    @abstractmethod
    def __getitem__(self, key: Union[int, slice]):
        """Get item(s) from the series by index or slice."""
        pass
    
    @abstractmethod
    def __iter__(self):
        """Return an iterator over the series values."""
        pass
    
    @abstractmethod
    def avg(self, window: Optional[int] = None) -> "DerivedSeries":
        """
        Create a derived series computing sliding window average.
        
        Args:
            window (int, optional): Window size. If None, uses all available data
                at each position (expanding window).
        
        Returns:
            DerivedSeries: A new series computing averages
        """
        pass
    
    @abstractmethod
    def sum(self, window: Optional[int] = None) -> "DerivedSeries":
        """
        Create a derived series computing sliding window sum.
        
        Args:
            window (int, optional): Window size. If None, uses all available data
                at each position (expanding window).
        
        Returns:
            DerivedSeries: A new series computing sums
        """
        pass
    
    @abstractmethod
    def cumsum(self) -> "DerivedSeries":
        """
        Create a derived series computing cumulative sum.
        
        Returns:
            DerivedSeries: A new series computing cumulative sums
        """
        pass
    
    @abstractmethod
    def __add__(self, other: Union["BaseSeries", int, float]) -> "DerivedSeries":
        """Add operation: self + other (element-wise or scalar)."""
        pass
    
    @abstractmethod
    def __radd__(self, other: Union[int, float]) -> "DerivedSeries":
        """Reverse add operation: other + self."""
        pass
    
    @abstractmethod
    def __sub__(self, other: Union["BaseSeries", int, float]) -> "DerivedSeries":
        """Subtract operation: self - other (element-wise or scalar)."""
        pass
    
    @abstractmethod
    def __rsub__(self, other: Union[int, float]) -> "DerivedSeries":
        """Reverse subtract operation: other - self."""
        pass
    
    @abstractmethod
    def __mul__(self, other: Union["BaseSeries", int, float]) -> "DerivedSeries":
        """Multiply operation: self * other (element-wise or scalar)."""
        pass
    
    @abstractmethod
    def __rmul__(self, other: Union[int, float]) -> "DerivedSeries":
        """Reverse multiply operation: other * self."""
        pass
    
    @abstractmethod
    def __truediv__(self, other: Union["BaseSeries", int, float]) -> "DerivedSeries":
        """Divide operation: self / other (element-wise or scalar)."""
        pass
    
    @abstractmethod
    def __rtruediv__(self, other: Union[int, float]) -> "DerivedSeries":
        """Reverse divide operation: other / self."""
        pass
    
    @abstractmethod
    def shift(self, periods: int = 1, fill_value: Optional[Union[int, float]] = None) -> "DerivedSeries":
        """
        Shift series by specified number of periods.
        
        Args:
            periods: Number of periods to shift. Positive shifts forward (down),
                negative shifts backward (up).
            fill_value: Value to use for filling. If None, uses NaN for float/timestamp,
                0 for int types.
        
        Returns:
            DerivedSeries: New series with shifted values
        """
        pass
    
    @abstractmethod
    def diff(self, periods: int = 1) -> "DerivedSeries":
        """
        Calculate the difference between current value and shifted value.
        
        Equivalent to: self - self.shift(periods)
        
        Args:
            periods: Periods to shift for calculating difference. Default is 1.
                Positive values compute forward differences.
        
        Returns:
            DerivedSeries: Difference series
        """
        pass
    

class DerivedSeries(BaseSeries):
    """
    Computed series that lazily evaluates from source series.
    
    DerivedSeries implements the computation graph mechanism, where derived metrics
    automatically update when their source data changes. Features include:
    
    - Lazy evaluation: computation happens only when values are accessed
    - Automatic cache invalidation when source data changes
    - Sliding window operations with cumsum optimization for O(n) performance
    - Arithmetic operations (+, -, *, /) with element-wise or scalar operations
    - Support for nested dependencies (derived from derived series)
    - Memory efficient: caches results but doesn't duplicate source data
    
    The class uses cumsum caching to optimize sliding window avg/sum operations:
    - First access: O(n) to compute cumsum cache
    - Subsequent window calculations: O(n) using cached cumsum
    
    Args:
        source (BaseSeries): Source series (can be Series or DerivedSeries)
        operation (Literal["avg", "sum", "cumsum", "add", "sub", "mul", "div"]): Operation type
        window (int, optional): Window size for avg/sum operations. None means
            expanding window (use all data up to current position).
        dtype (str, optional): Override dtype. If None, inherits from source.
        source2 (BaseSeries or float, optional): Second operand for binary operations
    
    Attributes:
        source (BaseSeries): The source series
        operation (str): Operation type
        window (Optional[int]): Window size
        source2: Second operand for arithmetic operations
        
    Example:
        >>> series = Series("float")
        >>> for i in range(10):
        ...     series.append(float(i))
        >>> 
        >>> # Create derived series
        >>> avg_series = series.avg(window=3)  # DerivedSeries
        >>> print(avg_series[4])  # (2+3+4)/3 = 3.0
        >>> 
        >>> # Arithmetic operations
        >>> series2 = series + 10  # Add scalar
        >>> series3 = series * series2  # Element-wise multiplication
    """
    
    def __init__(
        self,
        source: BaseSeries,
        operation: Literal["avg", "sum", "cumsum", "add", "sub", "mul", "div", "shift", "identity"],
        window: Optional[int] = None,
        dtype: Optional[str] = None,
        source2: Union[BaseSeries, int, float, None] = None,
        shift_periods: Optional[int] = None,
        shift_fill_value: Optional[Union[int, float]] = None,
        slice_obj: Optional[slice] = None
    ):
        """
        Initialize a DerivedSeries.
        
        Args:
            source: Source series to compute from
            operation: Operation type ("avg", "sum", "cumsum", "add", "sub", "mul", "div", "shift", "identity")
            window: Window size for avg/sum operations
            dtype: Override dtype (defaults to source.dtype)
            source2: Second operand for arithmetic operations
            shift_periods: Number of periods to shift (for shift operation)
            shift_fill_value: Fill value for shift operation
        
        Raises:
            ValueError: If operation is invalid or if cumsum is called with window
        """
        valid_ops = ["avg", "sum", "cumsum", "add", "sub", "mul", "div", "shift", "identity"]
        if operation not in valid_ops:
            raise ValueError(f"Unsupported operation: {operation}")
        
        if operation == "cumsum" and window is not None:
            raise ValueError("cumsum does not support window parameter")
        
        self.source = source
        self.operation = operation
        self.window = window
        self.source2 = source2
        self.shift_periods = shift_periods
        self.shift_fill_value = shift_fill_value
        self._dtype = dtype if dtype is not None else source.dtype
        self.slice_obj = slice_obj
        
        # Cache for computed results
        self._cache: Optional[List] = None
        self._cache_valid = False
        
        # Cache for cumsum (used to optimize avg/sum operations)
        self._cumsum_cache: Optional[List] = None
        self._cumsum_cache_valid = False
        
        # Track dependents for cache invalidation propagation
        self._dependents: List["DerivedSeries"] = []
        
        # Register this derived series as a dependent of the source(s)
        if isinstance(source, (Series, DerivedSeries)):
            source._dependents.append(self)
        
        # For binary operations, also register with source2 if it's a series
        if source2 is not None and isinstance(source2, (Series, DerivedSeries)):
            source2._dependents.append(self)
    
    @property
    def dtype(self) -> str:
        """Return the data type of values in this series."""
        return self._dtype
    
    def _invalidate_cache(self):
        """
        Recursively invalidate cache for this and all dependent series.
        
        Called when source data changes, propagating invalidation through
        the computation graph.
        """
        self._cache_valid = False
        self._cumsum_cache_valid = False
        
        # Recursively invalidate dependents
        for dependent in getattr(self, "_dependents", []):
            dependent._invalidate_cache()
    
    def _compute(self) -> List:
        """
        Perform lazy computation and return results.
        
        Computes the derived values from source data using the specified operation
        and window. Results are cached until source data changes.
        
        Returns:
            List: Computed values
        
        Time Complexity:
            - avg/sum with window: O(n) with cumsum optimization
            - cumsum: O(n)
            - arithmetic operations: O(n)
        """
        if self._cache_valid and self._cache is not None:
            # Apply slice to cached result if specified
            if self.slice_obj is not None:
                return self._cache[self.slice_obj]
            return self._cache
        
        # Get source data (triggers recursive computation if source is derived)
        source_data = list(self.source)
        
        # Compute based on operation
        if self.operation == "avg":
            result = self._compute_rolling_avg(source_data, self.window)
        elif self.operation == "sum":
            result = self._compute_rolling_sum(source_data, self.window)
        elif self.operation == "cumsum":
            result = self._compute_cumsum(source_data)
        elif self.operation in ["add", "sub", "mul", "div"]:
            result = self._compute_arithmetic(source_data, self.operation, self.source2)
        elif self.operation == "shift":
            result = self._compute_shift(source_data, self.shift_periods, self.shift_fill_value)
        elif self.operation == "identity":
            # Identity: pass through source data as-is (reduction happens on indexing)
            result = source_data
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
        
        # Cache the result (before slicing, so we cache the full computation)
        self._cache = result
        self._cache_valid = True
        
        # Apply slice if specified
        if self.slice_obj is not None:
            result = result[self.slice_obj]
        
        return result
    
    def _compute_cumsum(self, data: List) -> List:
        """
        Compute cumulative sum, skipping None values.
        
        Args:
            data: Input data list
            
        Returns:
            List of cumulative sums
            
        Time Complexity: O(n)
        """
        result = []
        cumsum = 0
        for val in data:
            if val is not None:
                cumsum += val
                result.append(cumsum)
            else:
                # Keep last cumsum or None if no cumsum yet
                result.append(cumsum if len(result) > 0 or cumsum != 0 else None)
        return result
    
    def _compute_arithmetic(self, data: List, operation: str, operand: Union[BaseSeries, int, float]) -> List:
        """
        Compute arithmetic operations (add, sub, mul, div).
        
        None values propagate: None op x = None
        
        Supports both element-wise operations (when operand is a Series) and
        scalar operations (when operand is a number).
        
        Args:
            data: Input data list from source
            operation: Operation type ("add", "sub", "mul", "div")
            operand: Second operand (can be Series/DerivedSeries or scalar)
            
        Returns:
            List of results
            
        Raises:
            ValueError: If operand lengths don't match for element-wise operations
            ZeroDivisionError: If division by zero occurs
            
        Time Complexity: O(n)
        """
        # Get operand data if it's a series
        if isinstance(operand, (Series, DerivedSeries)):
            operand_data = list(operand)
            # Check length compatibility
            if len(data) != len(operand_data):
                raise ValueError(
                    f"Length mismatch: source has {len(data)} elements, "
                    f"operand has {len(operand_data)} elements"
                )
            is_scalar = False
        else:
            # Scalar operand
            operand_data = operand
            is_scalar = True
        
        result = []
        for i, val in enumerate(data):
            operand_val = operand_data if is_scalar else operand_data[i]
            
            # None propagation: if either value is None, result is None
            if val is None or operand_val is None:
                result.append(None)
                continue
            
            if operation == "add":
                result.append(val + operand_val)
            elif operation == "sub":
                result.append(val - operand_val)
            elif operation == "mul":
                result.append(val * operand_val)
            elif operation == "div":
                if operand_val == 0:
                    raise ZeroDivisionError(f"Division by zero at index {i}")
                result.append(val / operand_val)
        
        return result
    
    def _compute_shift(
        self, 
        data: List, 
        periods: Optional[int], 
        fill_value: Optional[Union[int, float]]
    ) -> List:
        """
        Shift data by specified number of periods.
        
        Positive periods shift forward (down), negative shift backward (up).
        Similar to pandas shift() behavior. Uses None as default fill value.
        
        Args:
            data: Input data list
            periods: Number of periods to shift (default 1)
            fill_value: Value to use for filling. If None, uses None as default.
            
        Returns:
            List of shifted values
            
        Time Complexity: O(n)
        
        Example:
            data = [1, 2, 3, 4, 5]
            shift(1) -> [None, 1, 2, 3, 4]
            shift(-1) -> [2, 3, 4, 5, None]
            shift(2) -> [None, None, 1, 2, 3]
        """
        if periods is None:
            periods = 1
        
        if len(data) == 0:
            return []
        
        # Use None as default fill value
        if fill_value is None:
            fill_value = None
        
        result = []
        
        if periods > 0:
            # Shift forward (down): prepend fill values
            for _ in range(min(periods, len(data))):
                result.append(fill_value)
            for i in range(len(data) - periods):
                result.append(data[i])
        elif periods < 0:
            # Shift backward (up): append fill values
            abs_periods = abs(periods)
            for i in range(abs_periods, len(data)):
                result.append(data[i])
            for _ in range(min(abs_periods, len(data))):
                result.append(fill_value)
        else:
            # periods == 0: no shift
            result = data.copy()
        
        return result
    
    def _ensure_cumsum_cache(self, data: List) -> List:
        """
        Ensure cumsum cache exists for the given data.
        
        Used internally to optimize sliding window calculations.
        
        Args:
            data: Source data to compute cumsum from
            
        Returns:
            List: Cumulative sums
        """
        if not self._cumsum_cache_valid or self._cumsum_cache is None:
            self._cumsum_cache = self._compute_cumsum(data)
            self._cumsum_cache_valid = True
        return self._cumsum_cache
    
    def _compute_rolling_sum(self, data: List, window: Optional[int]) -> List:
        """
        Compute sliding window sum, excluding None values.
        
        For each position i, computes the sum of non-None values in the window.
        
        Args:
            data: Input data list
            window: Window size. If None, uses expanding window (all data up to i).
            
        Returns:
            List of rolling sums
            
        Time Complexity: O(n * window) - cannot use cumsum optimization with None values
        """
        if len(data) == 0:
            return []
        
        result = []
        for i in range(len(data)):
            if window is None:
                # Expanding window: sum from beginning to current position
                values = [data[j] for j in range(i + 1) if data[j] is not None]
            else:
                # Sliding window
                start = max(0, i - window + 1)
                values = [data[j] for j in range(start, i + 1) if data[j] is not None]
            
            if len(values) > 0:
                result.append(sum(values))
            else:
                result.append(None)
        
        return result
    
    def _compute_rolling_avg(self, data: List, window: Optional[int]) -> List:
        """
        Compute sliding window average, excluding None values.
        
        For each position i, computes the average of non-None values in the window.
        
        Args:
            data: Input data list
            window: Window size. If None, uses expanding window.
            
        Returns:
            List of rolling averages
            
        Time Complexity: O(n * window)
        """
        if len(data) == 0:
            return []
        
        result = []
        for i in range(len(data)):
            if window is None:
                # Expanding window
                values = [data[j] for j in range(i + 1) if data[j] is not None]
            else:
                # Sliding window
                start = max(0, i - window + 1)
                values = [data[j] for j in range(start, i + 1) if data[j] is not None]
            
            if len(values) > 0:
                result.append(sum(values) / len(values))
            else:
                result.append(None)
        
        return result
    
    def __len__(self) -> int:
        """Return the length of the derived series."""
        if self.slice_obj is not None:
            # Need to compute to know the sliced length
            computed = self._compute()
            return len(computed)
        return len(self.source)
    
    def __getitem__(self, key: Union[int, slice]):
        """
        Get item(s) from the derived series.
        
        Triggers lazy computation if cache is invalid. For slices, returns
        a new DerivedSeries with a sliced source. For identity operations
        with reduce, applies distributed reduction to the accessed value.
        
        Args:
            key: Index or slice object
            
        Returns:
            Union[float, int, Scalar, DerivedSeries]: Single value (or Scalar 
                for reduced values) for index, DerivedSeries for slice
        """
        if isinstance(key, slice):
            # Return a new DerivedSeries that references this DerivedSeries
            # to maintain the dependency chain for proper cache invalidation
            return DerivedSeries(
                source=self,
                operation="identity",
                window=None,
                dtype=self._dtype,
                source2=None,
                shift_periods=None,
                shift_fill_value=None,
                slice_obj=key
            )
        else:
            # Single index: compute and return value
            computed = self._compute()
            value = computed[key]
            
            return value
    
    def __iter__(self):
        """Iterate over computed values."""
        computed = self._compute()
        return iter(computed)
    
    def avg(self, window: Optional[int] = None) -> "DerivedSeries":
        """
        Create a derived series computing average from this series.
        
        Supports nested derivation (computing average of a derived series).
        
        Args:
            window: Window size for sliding window
            
        Returns:
            DerivedSeries: New derived series
        """
        return DerivedSeries(source=self, operation="avg", window=window)
    
    def sum(self, window: Optional[int] = None) -> "DerivedSeries":
        """
        Create a derived series computing sum from this series.
        
        Supports nested derivation (computing sum of a derived series).
        
        Args:
            window: Window size for sliding window
            
        Returns:
            DerivedSeries: New derived series
        """
        return DerivedSeries(source=self, operation="sum", window=window)
    
    def cumsum(self) -> "DerivedSeries":
        """
        Create a derived series computing cumulative sum from this series.
        
        Supports nested derivation.
        
        Returns:
            DerivedSeries: New derived series
        """
        return DerivedSeries(source=self, operation="cumsum", window=None)
    
    def __add__(self, other: Union[BaseSeries, int, float]) -> "DerivedSeries":
        """
        Add operation: self + other.
        
        Supports both element-wise addition (when other is a Series) and
        scalar addition (when other is a number).
        
        Args:
            other: Series or scalar to add
            
        Returns:
            DerivedSeries: New derived series with addition results
            
        Example:
            >>> s1 = Series("float")
            >>> for i in [1, 2, 3]:
            ...     s1.append(float(i))
            >>> s2 = s1 + 10  # Scalar addition
            >>> print(list(s2))  # [11.0, 12.0, 13.0]
        """
        return DerivedSeries(source=self, operation="add", source2=other)
    
    def __radd__(self, other: Union[int, float]) -> "DerivedSeries":
        """Reverse add: other + self (same as __add__ for addition)."""
        return self.__add__(other)
    
    def __sub__(self, other: Union[BaseSeries, int, float]) -> "DerivedSeries":
        """
        Subtract operation: self - other.
        
        Supports both element-wise subtraction (when other is a Series) and
        scalar subtraction (when other is a number).
        
        Args:
            other: Series or scalar to subtract
            
        Returns:
            DerivedSeries: New derived series with subtraction results
        """
        return DerivedSeries(source=self, operation="sub", source2=other)
    
    def __rsub__(self, other: Union[int, float]) -> "DerivedSeries":
        """
        Reverse subtract: other - self.
        
        Implemented as: DerivedSeries with scalar - self at each position.
        """
        # Create a temporary series with the scalar value repeated
        # Then subtract self from it
        return DerivedSeries(source=self, operation="sub", source2=other, dtype=self.dtype)._reverse_sub(other)
    
    def _reverse_sub(self, scalar: Union[int, float]) -> "DerivedSeries":
        """Helper for reverse subtraction."""
        # We need to create a derived series that computes: scalar - self[i]
        # This is implemented by negating and adding
        neg_self = DerivedSeries(source=self, operation="mul", source2=-1)
        return DerivedSeries(source=neg_self, operation="add", source2=scalar)
    
    def __mul__(self, other: Union[BaseSeries, int, float]) -> "DerivedSeries":
        """
        Multiply operation: self * other.
        
        Supports both element-wise multiplication (when other is a Series) and
        scalar multiplication (when other is a number).
        
        Args:
            other: Series or scalar to multiply
            
        Returns:
            DerivedSeries: New derived series with multiplication results
        """
        return DerivedSeries(source=self, operation="mul", source2=other)
    
    def __rmul__(self, other: Union[int, float]) -> "DerivedSeries":
        """Reverse multiply: other * self (same as __mul__ for multiplication)."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[BaseSeries, int, float]) -> "DerivedSeries":
        """
        Divide operation: self / other.
        
        Supports both element-wise division (when other is a Series) and
        scalar division (when other is a number).
        
        Args:
            other: Series or scalar to divide by
            
        Returns:
            DerivedSeries: New derived series with division results
            
        Raises:
            ZeroDivisionError: If division by zero occurs
        """
        return DerivedSeries(source=self, operation="div", source2=other)
    
    def __rtruediv__(self, other: Union[int, float]) -> "DerivedSeries":
        """
        Reverse divide: other / self.
        
        Implemented as: scalar / self[i] at each position.
        """
        return DerivedSeries(source=self, operation="div", source2=other, dtype=self.dtype)._reverse_div(other)
    
    def _reverse_div(self, scalar: Union[int, float]) -> "DerivedSeries":
        """Helper for reverse division: scalar / self."""
        # We need to create a derived series that computes: scalar / self[i]
        # This requires a special handling - we'll create an intermediate operation
        # For now, use reciprocal then multiply: scalar * (1 / self[i])
        reciprocal = DerivedSeries(source=self, operation="div", source2=1.0)  # 1 / self
        # Then multiply by scalar - but we need to swap the order
        # Actually, we can just create a custom computation
        # Let's add a helper that does scalar / source directly
        class ReverseDivSeries(DerivedSeries):
            def __init__(self, source, scalar):
                super().__init__(source=source, operation="div", source2=scalar)
                self._reverse_scalar = scalar
            
            def _compute_arithmetic(self, data, operation, operand):
                # Override to do scalar / data[i] instead of data[i] / scalar
                # Handle None propagation
                result = []
                for val in data:
                    if val is None:
                        result.append(None)
                    elif val == 0:
                        raise ZeroDivisionError(f"Division by zero")
                    else:
                        result.append(self._reverse_scalar / val)
                return result
        
        return ReverseDivSeries(source=self, scalar=scalar)
    
    def shift(self, periods: int = 1, fill_value: Optional[Union[int, float]] = None) -> "DerivedSeries":
        """
        Shift series by specified number of periods.
        
        Positive periods shift forward (down), negative shift backward (up).
        Follows pandas shift() semantics.
        
        Args:
            periods: Number of periods to shift. Default is 1.
                - Positive: shift forward (later values move down)
                - Negative: shift backward (earlier values move up)
                - Zero: no shift
            fill_value: Value to fill for missing positions. If None, uses
                NaN for float/timestamp types, 0 for int type.
        
        Returns:
            DerivedSeries: New derived series with shifted values
        
        Example:
            >>> series = Series("float")
            >>> for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ...     series.append(i)
            >>> 
            >>> # Shift forward by 1
            >>> shifted = series.shift(1)
            >>> print(list(shifted))  # [NaN, 1.0, 2.0, 3.0, 4.0]
            >>> 
            >>> # Shift backward by 1
            >>> shifted_back = series.shift(-1)
            >>> print(list(shifted_back))  # [2.0, 3.0, 4.0, 5.0, NaN]
            >>> 
            >>> # Custom fill value
            >>> shifted_zero = series.shift(2, fill_value=0)
            >>> print(list(shifted_zero))  # [0, 0, 1.0, 2.0, 3.0]
        """
        return DerivedSeries(
            source=self,
            operation="shift",
            shift_periods=periods,
            shift_fill_value=fill_value
        )
    
    def diff(self, periods: int = 1) -> "DerivedSeries":
        """
        Calculate first discrete difference.
        
        Computes the difference between the current element and an element
        `periods` positions earlier. Equivalent to: self - self.shift(periods)
        
        Args:
            periods: Periods to shift for calculating difference. Default is 1.
        
        Returns:
            DerivedSeries: New series with differences
        
        Example:
            >>> series = Series("float")
            >>> for i in [1.0, 3.0, 6.0, 10.0]:
            ...     series.append(i)
            >>> avg = series.avg(window=2)
            >>> diff = avg.diff()  # Difference of averaged values
        """
        return self - self.shift(periods)


class Series(list, BaseSeries):
    """
    Primary time series container storing actual data.
    
    Series extends Python's built-in list to provide a specialized container for
    tracking sequences of metrics with computation graph support. It inherits all 
    list operations (indexing, slicing, iteration) while adding:
    
    - Storage of actual numeric data (int, float, or timestamp)
    - Dependency tracking for derived series
    - Automatic cache invalidation on data changes
    - Creation of derived series via avg(), sum(), cumsum() methods
    - Automatic timestamp tracking via tick() method
    - Slice operations return Series objects (not lists) to preserve type
    
    When data is appended to a Series, all dependent DerivedSeries are notified
    to invalidate their caches, enabling automatic updates through the computation
    graph.
    
    Args:
        dtype (Literal["int", "float", "timestamp"]): Data type for the series.
            "int" stores integer values, "float" stores floating-point values,
            "timestamp" stores Unix timestamps as floats.
    
    Attributes:
        dtype (str): The data type of the series
        
    Example:
        >>> series = Series("float")
        >>> for i in range(5):
        ...     series.append(float(i))
        >>> 
        >>> # Create derived series
        >>> avg_series = series.avg(window=3)  # Returns DerivedSeries
        >>> print(avg_series[3])  # (1+2+3)/3 = 2.0
        >>> 
        >>> # Automatic update on append
        >>> series.append(5.0)
        >>> print(avg_series[4])  # (2+3+5)/3 = 3.33...
        >>> 
        >>> # Slicing returns a Series object
        >>> sliced = series[0:3]
        >>> print(type(sliced))  # <class 'Series'>
    """
    
    def __init__(
        self, 
        dtype: Literal["int", "float", "timestamp"], 
        metrics: Optional["Metrics"] = None,
        fill_value: Union[float, int, callable, None] = None,
        initial_value: Union[float, int, callable, None] = None,
        reduce: Optional[Literal["mean", "sum"]] = None
    ):
        """
        Initialize an empty Series with specified data type.
        
        Args:
            dtype: Data type for all values in this series
            metrics: Parent Metrics instance for index alignment.
                If None, series operates independently without index alignment.
            fill_value: Value to use when series has no value for a step.
                Can be a callable. Defaults to None (missing value).
            initial_value: Initial sentinel value added by initialize().
                Can be a callable. Defaults to None.
            reduce: Reduction strategy across ranks ("mean", "sum", or None)
        """
        super().__init__()
        self._dtype = dtype
        self._dependents: List[DerivedSeries] = []  # Track dependent DerivedSeries
        self._metrics = metrics  # Reference to parent Metrics for index alignment
        self._index_positions = {}  # Maps global index to list position
        
        # Store new parameters
        self._fill_value = fill_value
        self._initial_value = initial_value
        self._reduce = reduce
    
    @property
    def dtype(self) -> str:
        """Return the data type of values in this series."""
        return self._dtype
    
    def _invalidate_cache(self):
        """
        Recursively invalidate caches of all dependent DerivedSeries.
        
        Called internally when data is appended to propagate cache invalidation
        through the computation graph.
        """
        for dependent in self._dependents:
            dependent._invalidate_cache()
    
    def _get_fill_value(self):
        """Get the fill value (call if callable). Returns None by default."""
        if self._fill_value is None:
            return None
        if callable(self._fill_value):
            return self._fill_value()
        return self._fill_value
    
    def _get_initial_value(self):
        """Get the initial value (call if callable). Returns None by default."""
        if self._initial_value is None:
            return None
        if callable(self._initial_value):
            return self._initial_value()
        return self._initial_value
    
    def _get_process_group(self):
        """Get the process group from parent Metrics."""
        if self._metrics is not None:
            return self._metrics._process_group
        return None
    
    def __getitem__(self, key):
        """
        Get item(s) from the series.
        
        Supports both integer indexing and slicing. When slicing, returns a new
        Series object with the same dtype, preserving the Series interface.
        
        Args:
            key (int or slice): Index or slice object
            
        Returns:
            Union[int, float, Series]: Single value for integer index, or new
                Series object for slice
                
        Example:
            >>> series = Series("float")
            >>> for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ...     series.append(i)
            >>> print(series[0])  # 1.0
            >>> sliced = series[1:4]  # Returns Series
            >>> print(type(sliced))  # <class 'DerivedSeries'>
            >>> print(list(sliced))  # [2.0, 3.0, 4.0]
        """
        # If result is a slice, return a DerivedSeries to maintain dependency chain
        if isinstance(key, slice):
            return DerivedSeries(
                source=self,
                operation="identity",
                window=None,
                dtype=self.dtype,
                source2=None,
                shift_periods=None,
                shift_fill_value=None,
                slice_obj=key
            )
        
        # Otherwise, get the single value and return it
        result = super().__getitem__(key)
        return result
    
    def append(self, x: Union[int, float, None, Scalar, "torch.Tensor"]):
        """
        Append a value to the series.
        
        Constraint: len(series) <= len(index) + 1
        None is allowed as missing value.
        
        Args:
            x: Value to append. Can be:
               - int/float: Direct scalar value
               - None: Missing value (for timestamp dtype, means current time)
               - Scalar: Wrapper object, value will be extracted
               - torch.Tensor: GPU tensor, stored as-is to avoid CPU-GPU sync.
                 Will be converted to scalar in step() when distributed reduction
                 or logging occurs.
        
        Example:
            >>> metrics = Metrics()
            >>> metrics.new("loss", dtype="float")
            >>> metrics.initialize()
            >>> 
            >>> # Can append once (will be for next step)
            >>> metrics.loss.append(1.0)
            >>> # Or append a tensor (no sync until step())
            >>> metrics.loss.append(loss.detach())
            >>> 
            >>> # Call step() to finalize the step
            >>> metrics.step()
        """
        # Handle timestamp special case
        if x is None and self.dtype == "timestamp":
            x = time.time()
        
        # Extract value from Scalar if needed
        if isinstance(x, Scalar):
            x = x.value
        
        # Check if it's a tensor - store as-is to avoid sync
        # Conversion to scalar will happen in step() or when logging
        if isinstance(x, torch.Tensor):
            # Detach to avoid holding computation graph
            x = x.detach()
            # Don't call .item() here - defer to step() for batch sync
        elif x is not None:
            # Type conversion for non-tensor values (None stays as None)
            if self.dtype == "int":
                x = int(x)
            elif self.dtype == "float" or self.dtype == "timestamp":
                x = float(x)
        
        if self._metrics is not None:
            # Check constraint: len(series) <= len(index) + 1
            if len(self) > len(self._metrics._index):
                raise RuntimeError(
                    f"Cannot append to series: already appended {len(self)} values "
                    f"but only {len(self._metrics._index)} index steps exist. "
                    f"Constraint violated: len(series) > len(index). "
                    f"Call metrics.step() to advance the index."
                )
            
            # Fill skipped indices with None
            while len(self) < len(self._metrics._index):
                super().append(None)
        
        super().append(x)
        self._invalidate_cache()
    
    def _get_missing_value(self) -> Union[int, float]:
        """
        DEPRECATED: Use None instead.
        Get the appropriate missing value for this series dtype.
        
        Returns:
            Union[int, float]: Missing value (NaN for float/timestamp, 0 for int)
        """
        if self.dtype == "int":
            return 0
        else:
            # float or timestamp
            return float('nan')
    
    def tick(self) -> float:
        """
        Record the current timestamp to the series (timestamp dtype only).
        
        Convenience method for timestamp Series that appends time.time() without
        requiring an explicit argument. Equivalent to append(time.time()).
        Automatically invalidates dependent caches.
        
        Returns:
            float: The recorded timestamp value
            
        Raises:
            TypeError: If Series dtype is not "timestamp"
            
        Example:
            >>> import time
            >>> series = Series("timestamp")
            >>> t1 = series.tick()
            >>> time.sleep(0.1)
            >>> t2 = series.tick()
            >>> print(t2 > t1)  # True
        """
        if self.dtype != "timestamp":
            raise TypeError(f"tick() is only available for timestamp Series, got dtype={self.dtype}")
        
        timestamp = time.time()
        super().append(timestamp)
        
        # Invalidate caches of all dependent DerivedSeries
        self._invalidate_cache()
        
        return timestamp
    
    def avg(self, window: Optional[int] = None) -> DerivedSeries:
        """
        Create a derived series computing sliding window average.
        
        Returns a DerivedSeries that lazily computes averages with the specified
        window size. The derived series automatically updates when new values are
        appended to this series.
        
        Args:
            window (int, optional): Window size for sliding window. If None, uses
                expanding window (all data from start up to each position).
                
        Returns:
            DerivedSeries: A new derived series computing averages
            
        Example:
            >>> series = Series("float")
            >>> for i in range(5):
            ...     series.append(float(i))
            >>> 
            >>> # Create sliding average with window=3
            >>> avg_series = series.avg(window=3)
            >>> print(avg_series[3])  # (1+2+3)/3 = 2.0
            >>> print(avg_series[4])  # (2+3+4)/3 = 3.0
            >>> 
            >>> # Automatic update
            >>> series.append(5.0)
            >>> print(avg_series[5])  # (3+4+5)/3 = 4.0
        """
        return DerivedSeries(source=self, operation="avg", window=window)
    
    def sum(self, window: Optional[int] = None) -> DerivedSeries:
        """
        Create a derived series computing sliding window sum.
        
        Returns a DerivedSeries that lazily computes sums with the specified
        window size. The derived series automatically updates when new values are
        appended to this series.
        
        Args:
            window (int, optional): Window size for sliding window. If None, uses
                expanding window (all data from start up to each position).
                
        Returns:
            DerivedSeries: A new derived series computing sums
            
        Example:
            >>> series = Series("int")
            >>> for i in range(5):
            ...     series.append(i)
            >>> 
            >>> # Create sliding sum with window=3
            >>> sum_series = series.sum(window=3)
            >>> print(sum_series[3])  # 1+2+3 = 6
            >>> print(sum_series[4])  # 2+3+4 = 9
            >>> 
            >>> # Automatic update
            >>> series.append(5)
            >>> print(sum_series[5])  # 3+4+5 = 12
        """
        return DerivedSeries(source=self, operation="sum", window=window)
    
    def cumsum(self) -> DerivedSeries:
        """
        Create a derived series computing cumulative sum.
        
        Returns a DerivedSeries that lazily computes cumulative sums from the
        start of the series. The derived series automatically updates when new 
        values are appended to this series.
        
        Returns:
            DerivedSeries: A new derived series computing cumulative sums
            
        Example:
            >>> series = Series("int")
            >>> for i in [1, 2, 3, 4, 5]:
            ...     series.append(i)
            >>> 
            >>> cumsum_series = series.cumsum()
            >>> print(list(cumsum_series))  # [1, 3, 6, 10, 15]
            >>> 
            >>> # Automatic update
            >>> series.append(6)
            >>> print(cumsum_series[5])  # 21
        """
        return DerivedSeries(source=self, operation="cumsum", window=None)
    
    def __add__(self, other: Union[BaseSeries, int, float]) -> DerivedSeries:
        """
        Add operation: self + other.
        
        Supports both element-wise addition (when other is a Series/DerivedSeries)
        and scalar addition (when other is a number).
        
        Args:
            other: Series or scalar to add
            
        Returns:
            DerivedSeries: New derived series with addition results
            
        Example:
            >>> s1 = Series("float")
            >>> for i in [1.0, 2.0, 3.0]:
            ...     s1.append(i)
            >>> 
            >>> # Scalar addition
            >>> s2 = s1 + 10
            >>> print(list(s2))  # [11.0, 12.0, 13.0]
            >>> 
            >>> # Element-wise addition
            >>> s3 = Series("float")
            >>> for i in [0.5, 0.5, 0.5]:
            ...     s3.append(i)
            >>> s4 = s1 + s3
            >>> print(list(s4))  # [1.5, 2.5, 3.5]
        """
        return DerivedSeries(source=self, operation="add", source2=other)
    
    def __radd__(self, other: Union[int, float]) -> DerivedSeries:
        """Reverse add: other + self (same as __add__ for addition)."""
        return self.__add__(other)
    
    def __sub__(self, other: Union[BaseSeries, int, float]) -> DerivedSeries:
        """
        Subtract operation: self - other.
        
        Supports both element-wise subtraction and scalar subtraction.
        
        Args:
            other: Series or scalar to subtract
            
        Returns:
            DerivedSeries: New derived series with subtraction results
        """
        return DerivedSeries(source=self, operation="sub", source2=other)
    
    def __rsub__(self, other: Union[int, float]) -> DerivedSeries:
        """
        Reverse subtract: other - self.
        
        Args:
            other: Scalar value
            
        Returns:
            DerivedSeries: New derived series computing other - self[i]
        """
        # Implement as: -1 * self + other
        neg_self = DerivedSeries(source=self, operation="mul", source2=-1)
        return DerivedSeries(source=neg_self, operation="add", source2=other)
    
    def __mul__(self, other: Union[BaseSeries, int, float]) -> DerivedSeries:
        """
        Multiply operation: self * other.
        
        Supports both element-wise multiplication and scalar multiplication.
        
        Args:
            other: Series or scalar to multiply
            
        Returns:
            DerivedSeries: New derived series with multiplication results
            
        Example:
            >>> series = Series("float")
            >>> for i in [2.0, 3.0, 4.0]:
            ...     series.append(i)
            >>> doubled = series * 2
            >>> print(list(doubled))  # [4.0, 6.0, 8.0]
        """
        return DerivedSeries(source=self, operation="mul", source2=other)
    
    def __rmul__(self, other: Union[int, float]) -> DerivedSeries:
        """Reverse multiply: other * self (same as __mul__ for multiplication)."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[BaseSeries, int, float]) -> DerivedSeries:
        """
        Divide operation: self / other.
        
        Supports both element-wise division and scalar division.
        
        Args:
            other: Series or scalar to divide by
            
        Returns:
            DerivedSeries: New derived series with division results
            
        Raises:
            ZeroDivisionError: If division by zero occurs
            
        Example:
            >>> series = Series("float")
            >>> for i in [10.0, 20.0, 30.0]:
            ...     series.append(i)
            >>> halved = series / 2
            >>> print(list(halved))  # [5.0, 10.0, 15.0]
        """
        return DerivedSeries(source=self, operation="div", source2=other)
    
    def __rtruediv__(self, other: Union[int, float]) -> DerivedSeries:
        """
        Reverse divide: other / self.
        
        Args:
            other: Scalar value
            
        Returns:
            DerivedSeries: New derived series computing other / self[i]
            
        Raises:
            ZeroDivisionError: If any element in self is zero
        """
        # Create a special derived series for reverse division
        class ReverseDivSeries(DerivedSeries):
            def __init__(self, source, scalar):
                super().__init__(source=source, operation="div", source2=scalar)
                self._reverse_scalar = scalar
            
            def _compute_arithmetic(self, data, operation, operand):
                # Override to do scalar / data[i] instead of data[i] / scalar
                # Handle None propagation
                result = []
                for i, val in enumerate(data):
                    if val is None:
                        result.append(None)
                    elif val == 0:
                        raise ZeroDivisionError(f"Division by zero at index {i}")
                    else:
                        result.append(self._reverse_scalar / val)
                return result
        
        return ReverseDivSeries(source=self, scalar=other)
    
    def shift(self, periods: int = 1, fill_value: Optional[Union[int, float]] = None) -> DerivedSeries:
        """
        Shift series by specified number of periods.
        
        Positive periods shift forward (down), negative shift backward (up).
        Follows pandas shift() semantics. Returns a DerivedSeries that automatically
        updates when the source series changes.
        
        Args:
            periods: Number of periods to shift. Default is 1.
                - Positive: shift forward (later values move down, prepend fill)
                - Negative: shift backward (earlier values move up, append fill)
                - Zero: no shift (returns copy)
            fill_value: Value to fill for missing positions. If None:
                - float/timestamp types: uses float('nan')
                - int type: uses 0
        
        Returns:
            DerivedSeries: New derived series with shifted values
        
        Example:
            >>> series = Series("float")
            >>> for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ...     series.append(i)
            >>> 
            >>> # Shift forward by 1 (pandas-like behavior)
            >>> shifted = series.shift(1)
            >>> print(list(shifted))  # [NaN, 1.0, 2.0, 3.0, 4.0]
            >>> 
            >>> # Shift backward by 1
            >>> shifted_back = series.shift(-1)
            >>> print(list(shifted_back))  # [2.0, 3.0, 4.0, 5.0, NaN]
            >>> 
            >>> # Shift forward by 2
            >>> shifted_2 = series.shift(2)
            >>> print(list(shifted_2))  # [NaN, NaN, 1.0, 2.0, 3.0]
            >>> 
            >>> # Custom fill value
            >>> series_int = Series("int")
            >>> for i in [1, 2, 3, 4, 5]:
            ...     series_int.append(i)
            >>> shifted_fill = series_int.shift(1, fill_value=-1)
            >>> print(list(shifted_fill))  # [-1, 1, 2, 3, 4]
            >>> 
            >>> # Auto-update when source changes
            >>> series.append(6.0)
            >>> print(list(shifted))  # [NaN, 1.0, 2.0, 3.0, 4.0, 5.0]
        """
        return DerivedSeries(
            source=self,
            operation="shift",
            shift_periods=periods,
            shift_fill_value=fill_value
        )
    
    def diff(self, periods: int = 1) -> DerivedSeries:
        """
        Calculate first discrete difference.
        
        Computes the difference between the current element and an element
        `periods` positions earlier. This is a convenience method equivalent
        to: self - self.shift(periods)
        
        Commonly used for:
        - Computing rate of change
        - Calculating deltas between consecutive values
        - Time series analysis
        
        Args:
            periods: Periods to shift for calculating difference. Default is 1.
                - Positive: backward difference (current - previous)
                - Must be >= 1
        
        Returns:
            DerivedSeries: Series with differences. First `periods` elements
                will be NaN (for float/timestamp) or result of subtraction with
                fill value (for int with custom fill_value).
        
        Example:
            >>> series = Series("float")
            >>> for i in [2.0, 5.0, 9.0, 14.0, 20.0]:
            ...     series.append(i)
            >>> 
            >>> # Calculate differences
            >>> diff = series.diff()
            >>> # Result: [NaN, 3.0, 4.0, 5.0, 6.0]
            >>> print(list(diff)[1:])  # [3.0, 4.0, 5.0, 6.0]
            >>> 
            >>> # Larger shift
            >>> diff2 = series.diff(periods=2)
            >>> # Result: [NaN, NaN, 7.0, 9.0, 11.0]
            >>> 
            >>> # Use case: calculate rate of change
            >>> loss_change = metrics.loss.diff()
        """
        return self - self.shift(periods)


def _tensor_to_scalar(x):
    """
    Convert a value to Python scalar, handling tensors.
    
    This function defers .item() calls until they are absolutely needed
    (e.g., during distributed reduction or logging), avoiding unnecessary
    CPU-GPU synchronization during the hot path.
    
    Args:
        x: Value that may be a tensor, Scalar, or Python scalar
        
    Returns:
        Python scalar (int or float) or None
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        # This is where the sync happens - but only once per step
        return x.item()
    if isinstance(x, Scalar):
        return x.value
    return x


class Metrics:
    """
    Main metrics management system for training workflows with shared index.
    
    Metrics manages Series with a global shared index, similar to pandas DataFrame.
    All series within a Metrics instance share the same index, enabling aligned
    time-series operations. Features include:
    
    - Flattened storage: direct access via metrics.loss (no groups)
    - Shared global index for all series (default name: "step")
    - Index progression via step() method
    - Automatic alignment: series must sync with current index
    - Missing values handling when series don't have data for a step
    - Multiple data types (int, float, timestamp)
    - **Deferred tensor-to-scalar conversion**: tensors are converted to scalars
      only during step(), avoiding CPU-GPU sync in the training hot path
    
    Example:
        >>> metrics = Metrics()
        >>> # Create metrics
        >>> metrics.new("loss", dtype="float")
        >>> metrics.new("tokens", dtype="int")
        >>> 
        >>> # Record values with index progression
        >>> for i in range(100):
        ...     metrics.step()  # Increment index
        ...     metrics.loss.append(compute_loss())
        ...     metrics.tokens.append(batch_tokens)
        >>> 
        >>> # All series share the same index
        >>> total_tokens = metrics.tokens.cumsum()
        >>> tokens_per_step = total_tokens.diff()
    """
    
    def __init__(self, index_name: str = "step", process_group: Optional[dist.ProcessGroup] = None):
        """
        Initialize an empty Metrics manager with shared index.
        
        Args:
            index_name: Name for the shared index. Defaults to "step".
            process_group: Default process group for all series.
                Individual series can override this. Defaults to None (uses default group).
        """
        self._series = {}  # metric name -> Series instance mapping
        self._index = []  # Global index values
        self._index_name = index_name
        self._current_index = 0
        self._loggers: List[Logger] = []  # List of logger instances
        self._logger_proxy = LoggerProxy(self._loggers)
        self._process_group = process_group
        self._derived_series = {}  # Track derived series for debugging
    
    @property
    def logger(self) -> LoggerProxy:
        """
        Get the logger proxy for broadcasting operations to all loggers.
        
        Returns:
            LoggerProxy: Proxy object that broadcasts to all loggers
        """
        return self._logger_proxy
    
    def add_logger(self, logger: Logger):
        """
        Add a logger to this Metrics instance.
        
        Args:
            logger: Logger instance to add
            
        Example:
            >>> metrics = Metrics()
            >>> logger1 = Logger("main", [TensorBoardBackend("runs/exp1")])
            >>> logger2 = Logger("csv", [CSVBackend("metrics.csv")])
            >>> metrics.add_logger(logger1)
            >>> metrics.add_logger(logger2)
        """
        self._loggers.append(logger)
    
    def write_logs(self, global_step: int):
        """
        Write all tracked metrics to all loggers.
        
        Args:
            global_step: User-provided step, generally the global step.
        
        Example:
            >>> metrics.step()
            >>> metrics.loss.append(0.5)
            >>> metrics.write_logs(global_step=100)  # Writes to all backends
        """
        if len(self._index) == 0:
            return
        
        for logger in self._loggers:
            logger.write(global_step)
    
    def new(
        self, 
        name: str, 
        dtype: Literal["int", "float", "timestamp"],
        fill_value: Union[float, int, callable, None] = None,
        initial_value: Union[float, int, callable, None] = None,
        reduce: Optional[Literal["mean", "sum"]] = None
    ):
        """
        Create a new metric Series with distributed reduction support.
        
        Args:
            name: Name of the metric
            dtype: Data type ("int", "float", "timestamp")
            fill_value: Value for missing steps. None means missing value.
                Can be a callable. Example: fill_value=0
            initial_value: Initial sentinel value for initialize().
                Can be a callable. Example: initial_value=lambda: time.time()
            reduce: Reduction strategy ("mean", "sum", or None).
                - "mean": Average across ranks (None values excluded)
                - "sum": Sum across ranks (None values excluded)
                - None: No reduction (use local value)
        
        Example:
            >>> metrics = Metrics()
            >>> metrics.new("loss", dtype="float", reduce="mean")
            >>> metrics.new("tokens", dtype="int", reduce="sum")
        """
        series = Series(dtype, metrics=self, fill_value=fill_value, 
                       initial_value=initial_value, reduce=reduce)
        self._series[name] = series
    
    def register_derived(self, name: str, derived_series: "DerivedSeries"):
        """
        Register a derived series for debugging visibility.
        
        Args:
            name: Name for this derived series
            derived_series: The DerivedSeries instance
            
        Example:
            >>> loss_avg = metrics.loss.avg(window=20)
            >>> metrics.register_derived("loss_avg_20", loss_avg)
        """
        self._derived_series[name] = derived_series
    
    def get_world_size(self, process_group: Optional[dist.ProcessGroup] = None) -> int:
        """
        Get the world size for a process group.
        
        Args:
            process_group: Process group to query. If None, uses default group.
        
        Returns:
            int: Number of processes in the group. Returns 1 if not distributed.
        """
        if dist.is_available() and dist.is_initialized():
            if process_group is not None:
                return dist.get_world_size(process_group)
            return dist.get_world_size()
        return 1
    
    def get_rank(self, process_group: Optional[dist.ProcessGroup] = None) -> int:
        """
        Get the current rank in a process group.
        
        Args:
            process_group: Process group to query. If None, uses default group.
        
        Returns:
            int: Process rank. Returns 0 if not distributed.
        """
        if dist.is_available() and dist.is_initialized():
            if process_group is not None:
                return dist.get_rank(process_group)
            return dist.get_rank()
        return 0
    
    def initialize(self):
        """
        Initialize all series with sentinel values.
        
        Adds an initial value to all series and increments the index.
        Should be called once after all series are created with new().
        
        Example:
            >>> metrics = Metrics()
            >>> metrics.new("loss", dtype="float", initial_value=0.0)
            >>> metrics.new("tokens", dtype="int", initial_value=0)
            >>> metrics.initialize()
            >>> assert len(metrics._index) == 1
            >>> assert len(metrics.loss) == 1
        """
        # Increment index first
        self._index.append(self._current_index)
        self._current_index += 1
        
        # Add initial value to all series
        for name, series in self._series.items():
            value = series._get_initial_value()
            
            # Directly append to bypass index checks
            super(Series, series).append(value)
            series._index_positions[self._index[-1]] = len(series) - 1
    
    def step(self, value: Optional[int] = None):
        """
        Execute computation for current step - like TensorFlow's session.run().
        
        Process:
        1. Collects current step values from all series
        2. For series with reduce, gathers from all ranks using Metrics process_group
        3. Reduces (excluding None) and replaces local value
        4. Fills missing values with fill_value
        5. Increments the global index
        
        All series share the same process_group from Metrics.
        
        Args:
            value: Optional custom index value. If None, uses auto-increment.
        
        Example:
            >>> # Each rank appends local loss
            >>> metrics.loss.append(local_loss)
            >>> 
            >>> # step() performs distributed reduction
            >>> metrics.step()
            >>> # metrics.loss[-1] now contains mean across all ranks
        """
        # Prepare next index value
        if value is None:
            next_index = self._current_index
            self._current_index += 1
        else:
            next_index = value
            self._current_index = value + 1
        
        current_index_pos = len(self._index)
        
        if dist.is_available() and dist.is_initialized():
            # === Distributed mode ===
            
            # All series use the same process_group from Metrics
            pg = self._process_group
            
            # 1. Collect local values from all series
            # Convert tensors to scalars here - this is where sync happens (once per step)
            local_values = {}
            for name, series in self._series.items():
                if len(series) > current_index_pos:
                    raw_val = series[current_index_pos]
                    # Convert tensor to scalar (deferred sync point)
                    local_values[name] = _tensor_to_scalar(raw_val)
                else:
                    local_values[name] = None
            
            # 2. Gather all values from all ranks
            world_size = self.get_world_size(pg)
            all_values = [None] * world_size
            dist.all_gather_object(all_values, local_values, group=pg)
            
            # 3. Process each series
            for name, series in self._series.items():
                if series._reduce is not None:
                    # Collect non-None values from all ranks
                    rank_values = []
                    for rank_data in all_values:
                        val = rank_data[name]
                        if val is not None:
                            rank_values.append(val)
                    
                    # Reduce (excluding None)
                    if len(rank_values) > 0:
                        if series._reduce == "mean":
                            reduced_value = sum(rank_values) / len(rank_values)
                        elif series._reduce == "sum":
                            reduced_value = sum(rank_values)
                        else:
                            reduced_value = rank_values[0]
                    else:
                        # All ranks have None
                        reduced_value = None
                    
                    # Replace or append the reduced value
                    if len(series) > current_index_pos:
                        series[current_index_pos] = reduced_value
                    else:
                        super(Series, series).append(reduced_value)
                        
                else:
                    # No reduction needed
                    if len(series) <= current_index_pos:
                        # Missing value, fill with fill_value
                        fill_val = series._get_fill_value()
                        super(Series, series).append(fill_val)
                    # else: keep existing local value
                
                # Update index position mapping
                series._index_positions[next_index] = current_index_pos
                
                # Invalidate caches
                series._invalidate_cache()
        
        else:
            # === Single process mode ===
            for name, series in self._series.items():
                if len(series) > current_index_pos:
                    # Convert any tensor values to scalars (deferred sync point)
                    raw_val = series[current_index_pos]
                    scalar_val = _tensor_to_scalar(raw_val)
                    if isinstance(raw_val, torch.Tensor):
                        # Replace tensor with scalar value
                        series[current_index_pos] = scalar_val
                elif len(series) <= current_index_pos:
                    fill_val = series._get_fill_value()
                    super(Series, series).append(fill_val)
                
                series._index_positions[next_index] = current_index_pos
                series._invalidate_cache()
        
        # Increment the global index
        self._index.append(next_index)
    
    def __getattr__(self, name: str) -> Series:
        """
        Enable dynamic access to Series metrics.
        
        Allows accessing Series via dot notation (e.g., metrics.loss).
        
        Args:
            name: The name of the Series to access
            
        Returns:
            Series: The requested Series instance
            
        Raises:
            AttributeError: If the Series name is not found
            
        Example:
            >>> metrics = Metrics()
            >>> metrics.new("loss", dtype="float")
            >>> metrics.new("lr", dtype="float")
            >>> 
            >>> # Direct access
            >>> metrics.loss.append(0.5)
            >>> metrics.lr.append(0.001)
        """
        if name.startswith("_"):
            # Avoid conflicts with private attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if name not in self._series:
            raise AttributeError(f"Series '{name}' not found in Metrics")
        
        return self._series[name]
    
    def print_summary(self, last_n: int = 5, rank: Optional[int] = None, 
                      show_derived: bool = True):
        """
        Print a summary table of all metrics for debugging.
        
        Displays the latest values of all series in a formatted table.
        
        Args:
            last_n: Number of latest values to show for each series (default: 5)
            rank: Current rank for display. If None, doesn't show rank info.
            show_derived: Whether to show derived series (default: True)
        
        Example:
            >>> metrics.print_summary(last_n=3)
            >>> metrics.print_summary(last_n=5, rank=0, show_derived=True)
        """
        # Header
        if rank is not None:
            print(f"\n{'=' * 80}")
            print(f"METRICS SUMMARY (Rank {rank})")
        else:
            print(f"\n{'=' * 80}")
            print(f"METRICS SUMMARY")
        print(f"{'=' * 80}")
        
        # Index info
        print(f"Index: {self._index_name}, Current: {self._current_index - 1}, Total steps: {len(self._index)}")
        print(f"Base series: {len(self._series)}, Derived series: {len(self._derived_series)}")
        print()
        
        # Base series table
        if len(self._series) > 0:
            print("BASE SERIES:")
            print(f"{'Name':<20} {'Type':<10} {'Reduce':<10} {'Len':<6} {'Latest Values'}")
            print("-" * 80)
            
            for name, series in sorted(self._series.items()):
                dtype = series.dtype
                reduce_op = series._reduce if series._reduce else "None"
                length = len(series)
                latest_str = self._format_values(series, length, last_n)
                print(f"{name:<20} {dtype:<10} {reduce_op:<10} {length:<6} {latest_str}")
        
        # Derived series table
        if show_derived and len(self._derived_series) > 0:
            print()
            print("DERIVED SERIES:")
            print(f"{'Name':<20} {'Operation':<20} {'Len':<6} {'Latest Values'}")
            print("-" * 80)
            
            for name, derived in sorted(self._derived_series.items()):
                operation = derived.operation
                if derived.window is not None:
                    op_str = f"{operation}(w={derived.window})"
                else:
                    op_str = operation
                length = len(derived)
                latest_str = self._format_values(derived, length, last_n)
                print(f"{name:<20} {op_str:<20} {length:<6} {latest_str}")
        
        print(f"{'=' * 80}\n")
    
    def _format_values(self, series, length: int, last_n: int) -> str:
        """Helper to format latest values."""
        if length == 0:
            return "[]"
        
        start_idx = max(0, length - last_n)
        values = []
        for i in range(start_idx, length):
            val = series[i]
            if val is None:
                values.append("None")
            elif isinstance(val, float):
                values.append(f"{val:.3f}")
            else:
                values.append(str(val))
        return f"[{', '.join(values)}]"
