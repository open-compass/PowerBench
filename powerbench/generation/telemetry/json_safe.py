from __future__ import annotations

import json
from typing import Any


def json_default(o: Any) -> Any:
    """Convert common non-JSON-native scalars (numpy/pandas) for json.dumps."""
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    # pandas Timestamp / numpy generic .item()
    if hasattr(o, "isoformat") and callable(getattr(o, "isoformat")):
        try:
            return o.isoformat(sep=" ")
        except TypeError:
            return o.isoformat()
    if hasattr(o, "item") and callable(getattr(o, "item")):
        try:
            return o.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def json_dumps(obj: Any, *, ensure_ascii: bool = False, indent: int | None = 2, **kwargs: Any) -> str:
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, default=json_default, **kwargs)
