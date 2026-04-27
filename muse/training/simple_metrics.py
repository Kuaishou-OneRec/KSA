"""
Simple Metrics System for Training.

A lightweight alternative to the full Metrics system, designed for better performance
with minimal overhead. Uses simple dictionaries and accumulators instead of complex
lazy evaluation chains.
"""
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any
import torch.distributed as dist


class SimpleMetrics:
    """
    Simple metrics tracker that accumulates values and writes to TensorBoard.
    
    This is a lightweight replacement for the complex Metrics class,
    optimized for training performance.
    """
    
    def __init__(self, gradient_accumulation_steps: int = 1, logging_per_step: int = 1):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_per_step = logging_per_step
        
        # Accumulators for averaging
        self._accumulators: Dict[str, List[float]] = defaultdict(list)
        self._last_values: Dict[str, float] = {}
        
        # Step tracking
        self._micro_step = 0
        self._global_step = 0
        self._start_time = time.time()
        self._last_log_time = time.time()
        
        # TensorBoard writer (set externally)
        self.tb_writer = None
        
    def set_tb_writer(self, tb_writer):
        """Set the TensorBoard writer."""
        self.tb_writer = tb_writer
        
    def append(self, name: str, value: float):
        """Append a value to a metric accumulator."""
        if hasattr(value, 'item'):
            value = value.item()
        self._accumulators[name].append(value)
        self._last_values[name] = value
        
    def get_last(self, name: str, default: float = 0.0) -> float:
        """Get the last value of a metric."""
        return self._last_values.get(name, default)
    
    def get_avg(self, name: str, default: float = 0.0) -> float:
        """Get the average of accumulated values for a metric."""
        values = self._accumulators.get(name, [])
        if not values:
            return default
        return sum(values) / len(values)
    
    def step(self):
        """Advance micro step counter."""
        self._micro_step += 1
        if self._micro_step % self.gradient_accumulation_steps == 0:
            self._global_step += 1
            
    def is_gradient_accumulation_boundary(self) -> bool:
        """Check if at gradient accumulation boundary."""
        return (
            self._micro_step > 0 and
            self._micro_step % self.gradient_accumulation_steps == 0
        )
    
    def should_logging(self) -> bool:
        """Check if should log at this step."""
        return (self._global_step > 0 and 
                self._global_step % self.logging_per_step == 0 and
                self.is_gradient_accumulation_boundary())
    
    @property
    def global_step(self) -> int:
        return self._global_step
    
    @property
    def micro_step(self) -> int:
        return self._micro_step
        
    def write_logs(self, global_step: Optional[int] = None):
        """Write accumulated metrics to TensorBoard and reset accumulators."""
        if global_step is None:
            global_step = self._global_step
            
        if self.tb_writer is None or dist.get_rank() != 0:
            self._reset_accumulators()
            return
            
        current_time = time.time()
        seconds_per_step = (current_time - self._last_log_time) / self.logging_per_step
        self._last_log_time = current_time
        
        # Write all accumulated metrics
        for name, values in self._accumulators.items():
            if values:
                avg_value = sum(values) / len(values)
                # Determine group based on name prefix
                if name.startswith("training/") or name.startswith("metrics/") or name.startswith("perf/"):
                    tag = name
                elif name in ["lm_loss", "codebook_loss", "commitment_loss",
                              "learning_rate", "grad_norm"]:
                    tag = f"training/{name}"
                elif name.startswith("perplexity") or name.startswith("codebook_usage") or \
                     name.startswith("avg_") or name.startswith("video_") or name.startswith("combined_"):
                    tag = f"metrics/{name}"
                else:
                    tag = f"training/{name}"
                    
                self.tb_writer.add_scalar(tag, avg_value, global_step=global_step, new_style=True)
        
        # Write performance metrics
        self.tb_writer.add_scalar("perf/seconds_per_step", seconds_per_step, 
                                   global_step=global_step, new_style=True)
        
        # Reset accumulators
        self._reset_accumulators()
        
    def _reset_accumulators(self):
        """Reset all accumulators after logging."""
        self._accumulators.clear()
        
    def log_scalar(self, name: str, value: float, global_step: Optional[int] = None):
        """Directly log a scalar value to TensorBoard."""
        if self.tb_writer is None or dist.get_rank() != 0:
            return
        if global_step is None:
            global_step = self._global_step
        if hasattr(value, 'item'):
            value = value.item()
        self.tb_writer.add_scalar(name, value, global_step=global_step, new_style=True)


def create_simple_metrics(args, tb_writer=None) -> SimpleMetrics:
    """
    Factory function to create a SimpleMetrics instance.
    
    Args:
        args: Argument namespace with gradient_accumulation_steps and logging_per_step
        tb_writer: Optional TensorBoard SummaryWriter
        
    Returns:
        Configured SimpleMetrics instance
    """
    metrics = SimpleMetrics(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_per_step=args.logging_per_step
    )
    if tb_writer is not None:
        metrics.set_tb_writer(tb_writer)
    return metrics
