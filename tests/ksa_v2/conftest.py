"""Keep imports and all pytest/multiprocessing temporary files in this checkout."""

import os
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "examples/ksa_v2"
KERNEL_DIR = REPO_ROOT / "summary_attention_kernel/ksa_v2"
sys.path[:0] = [str(REPO_ROOT), str(ROOT / "tools"), str(KERNEL_DIR / "v0.1")]
scratch = Path(os.environ.get("TMPDIR", REPO_ROOT / ".tmp"))
scratch.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(scratch)
tempfile.tempdir = str(scratch)
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "output/test_cache/huggingface"))
os.environ.setdefault("TRITON_CACHE_DIR", str(REPO_ROOT / "output/test_cache/triton"))
