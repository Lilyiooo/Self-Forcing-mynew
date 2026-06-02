import json
import os
import time
from typing import Any

import torch
import torch.distributed as dist


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _to_jsonable(value.item())
        return list(value.shape)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    return str(value)


class CacheDebugLogger:
    """Small JSONL logger for train/inference cache event tracing."""

    def __init__(self, path: str, enabled: bool = True, print_events: bool = True):
        self.enabled = enabled
        self.print_events = print_events
        self.rank = _rank()
        self.path = path.format(rank=self.rank) if path else ""
        if self.enabled and self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

    def log(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        record = {
            "time": time.time(),
            "rank": self.rank,
            "event": event,
            **{k: _to_jsonable(v) for k, v in fields.items()},
        }
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        if self.print_events and self.rank == 0:
            print(f"[CacheDebug] {event}: {record}")


def make_cache_debug_logger(config, default_path: str | None = None) -> CacheDebugLogger | None:
    het_cfg = getattr(config, "heterogeneous_cache", config)
    if not getattr(het_cfg, "debug_log_cache", False):
        return None
    path = getattr(het_cfg, "debug_log_path", None) or default_path
    print_events = getattr(het_cfg, "debug_log_print", True)
    return CacheDebugLogger(path=path, enabled=True, print_events=print_events)
