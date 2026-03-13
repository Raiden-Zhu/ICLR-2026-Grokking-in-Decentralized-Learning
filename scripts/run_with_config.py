#!/usr/bin/env python3
"""Run training from a YAML config without modifying training logic."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _load_config(config_path):
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_override(raw):
    if "=" not in raw:
        raise ValueError(f"Invalid override '{raw}'. Expected key=value format.")
    key, raw_value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override '{raw}'. Key cannot be empty.")
    value = yaml.safe_load(raw_value)
    return key, value


def _apply_overrides(cfg, override_items):
    if not override_items:
        return cfg

    cfg = dict(cfg)
    cfg.setdefault("run", {})
    cfg.setdefault("env", {})
    cfg.setdefault("args", {})

    for item in override_items:
        key, value = _parse_override(item)
        if key.startswith("run."):
            cfg["run"][key[4:]] = value
        elif key.startswith("env."):
            cfg["env"][key[4:]] = value
        elif key.startswith("args."):
            cfg["args"][key[5:]] = value
        else:
            # Bare key defaults to training args for convenience.
            cfg["args"][key] = value
    return cfg


def _to_cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_command(python_cmd, entry_path, args_dict, r_start=None):
    cmd = [python_cmd, str(entry_path)]
    for key, value in args_dict.items():
        if key == "r_starts":
            continue
        if key == "r_start" and r_start is not None:
            value = r_start
        cmd.extend([f"--{key}", _to_cli_value(value)])
    if "r_start" not in args_dict and r_start is not None:
        cmd.extend(["--r_start", _to_cli_value(r_start)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch main_multi_GPU.py from YAML config")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values. Format: key=value or section.key=value",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command only")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_root = Path(__file__).resolve().parent.parent

    cfg = _apply_overrides(_load_config(config_path), args.set)

    run_cfg = cfg.get("run", {})
    env_cfg = cfg.get("env", {})
    arg_cfg = cfg.get("args", {})

    python_cmd = run_cfg.get("python", sys.executable)
    entry = run_cfg.get("entry", "main_multi_GPU.py")
    entry_path = (repo_root / entry).resolve()

    if not entry_path.exists():
        raise FileNotFoundError(f"Entry script not found: {entry_path}")

    run_env = os.environ.copy()
    for key, value in env_cfg.items():
        if key not in run_env or not run_env[key]:
            run_env[key] = str(value)

    r_starts = arg_cfg.get("r_starts")
    if r_starts is None:
        r_starts = [arg_cfg.get("r_start")]
    if not isinstance(r_starts, list):
        r_starts = [r_starts]

    for r_start in r_starts:
        cmd = _build_command(python_cmd, entry_path, arg_cfg, r_start=r_start)
        print("[Run]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(repo_root), env=run_env, check=True)


if __name__ == "__main__":
    main()
