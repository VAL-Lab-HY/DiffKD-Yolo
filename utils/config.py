from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, opts):
    """Apply dotted ``key=value`` overrides, e.g. ``train.epochs=100 teacher.input_size=512``."""
    for opt in opts or []:
        key, sep, value = opt.partition("=")
        if not sep:
            raise ValueError(f"Override '{opt}' must have the form key=value")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
