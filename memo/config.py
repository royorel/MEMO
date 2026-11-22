"""Scene configs.

A config may name a parent config in `_base_`, a path relative to itself.
Loading resolves the chain from the root down and deep-merges each child over
its parent, so a scene config holds only what differs from its dataset's
base, and a dataset base only what differs from `configs/base.yaml`.
"""

from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path) -> dict:
    """The resolved config at `path`, with its `_base_` chain merged in."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("_base_", None)
    return deep_merge(load_config(path.parent / parent), cfg) if parent else cfg
