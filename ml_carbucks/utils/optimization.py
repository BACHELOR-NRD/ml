import datetime as dt
from typing import Optional


def get_runtime(title: str, override: Optional[str] = None) -> str:
    if override is not None:
        return override

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime = f"{timestamp}_{title}"

    return runtime
