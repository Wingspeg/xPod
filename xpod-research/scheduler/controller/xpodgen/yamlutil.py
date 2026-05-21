import math
import logging
import re
from typing import Any, List

logger = logging.getLogger(__name__)


_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def yaml_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return "null"
        return str(v)
    s = str(v)
    if s == "":
        return '""'
    needs_quote = False
    if _NUMERIC_RE.match(s):
        needs_quote = True
    for ch in s:
        if ch in ":\n#{}[]," or ch.isspace():
            needs_quote = True
            break
    if s.lower() in {"null", "true", "false", "yes", "no", "~"}:
        needs_quote = True
    if s[0] in "-?@&*!%|>":
        needs_quote = True
    if not needs_quote:
        return s
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def to_yaml(obj: Any, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return f"{sp}{{}}"
        lines: List[str] = []
        for k, v in obj.items():
            if isinstance(v, list) and len(v) == 0:
                lines.append(f"{sp}{k}: []")
            elif isinstance(v, dict) and len(v) == 0:
                lines.append(f"{sp}{k}: {{}}")
            elif isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(to_yaml(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return f"{sp}[]"
        lines = []
        for it in obj:
            if isinstance(it, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(to_yaml(it, indent + 1))
            else:
                lines.append(f"{sp}- {yaml_scalar(it)}")
        return "\n".join(lines)
    return f"{sp}{yaml_scalar(obj)}"
