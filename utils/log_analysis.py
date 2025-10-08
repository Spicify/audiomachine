import re
import gzip
from collections import Counter
from typing import Dict, Any

from utils.s3_utils import s3_get_bytes


def parse_log_bytes(data: bytes) -> Dict[str, Any]:
    text = data.decode("utf-8", "ignore")
    lines = text.splitlines()
    levels: Counter[str] = Counter()
    errors: list[str] = []
    mem_usage: list[float] = []
    for line in lines:
        if "[ERROR]" in line:
            errors.append(line)
        if "[DIAG]" in line and "Memory" in line:
            m = re.search(r"Memory usage: ([0-9.]+)", line)
            if m:
                try:
                    mem_usage.append(float(m.group(1)))
                except Exception:
                    pass
        for lvl in ["INFO", "DEBUG", "ERROR", "DIAG", "UI"]:
            if f"[{lvl}]" in line:
                levels[lvl] += 1
    return {
        "levels": dict(levels),
        "errors": errors,
        "avg_mem": round(sum(mem_usage) / len(mem_usage), 2) if mem_usage else None,
    }


def load_log_summary(key: str) -> Dict[str, Any]:
    data = s3_get_bytes(key)
    if not data:
        return {"levels": {}, "errors": [], "avg_mem": None}
    if key.endswith(".gz"):
        data = gzip.decompress(data)
    return parse_log_bytes(data)


def bundle_logs_as_zip(keys: list[str]) -> bytes:
    from io import BytesIO
    import zipfile

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for k in keys:
            content = s3_get_bytes(k) or b""
            z.writestr(k.split("/")[-1], content)
    buf.seek(0)
    return buf.getvalue()
