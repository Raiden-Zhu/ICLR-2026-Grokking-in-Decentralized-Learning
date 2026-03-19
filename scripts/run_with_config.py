#!/usr/bin/env python3
"""Run training from a YAML config without modifying training logic."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config_validation import ALIAS_KEYS, collect_alias_deprecation_warnings, validate_training_kwargs


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
        return cfg, []

    cfg = dict(cfg)
    cfg.setdefault("run", {})
    cfg.setdefault("env", {})
    cfg.setdefault("args", {})
    deprecated_aliases = []

    for item in override_items:
        key, value = _parse_override(item)
        if key.startswith("run."):
            cfg["run"][key[4:]] = value
        elif key.startswith("env."):
            cfg["env"][key[4:]] = value
        elif key.startswith("args."):
            arg_key = key[5:]
            if arg_key in ALIAS_KEYS:
                deprecated_aliases.append(arg_key)
            cfg["args"][ALIAS_KEYS.get(arg_key, arg_key)] = value
        else:
            if key in ALIAS_KEYS:
                deprecated_aliases.append(key)
            cfg["args"][ALIAS_KEYS.get(key, key)] = value
    return cfg, deprecated_aliases


def _to_cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_command(python_cmd, entry_path, args_dict, r_start=None):
    cmd = [python_cmd, str(entry_path)]
    for key, value in args_dict.items():
        if value is None or key == "r_starts":
            continue
        if key == "r_start" and r_start is not None:
            value = r_start
        cmd.extend([f"--{key}", _to_cli_value(value)])
    if "r_start" not in args_dict and r_start is not None:
        cmd.extend(["--r_start", _to_cli_value(r_start)])
    return cmd


def _normalize_arg_config(arg_cfg):
    arg_cfg = dict(arg_cfg)
    r_starts = arg_cfg.pop("r_starts", None)
    deprecation_warnings = collect_alias_deprecation_warnings(arg_cfg)
    normalized_args = validate_training_kwargs(arg_cfg, require_all=False)
    if r_starts is not None:
        normalized_args["r_starts"] = r_starts
    return normalized_args, deprecation_warnings


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

    cfg, deprecated_override_aliases = _apply_overrides(_load_config(config_path), args.set)

    run_cfg = cfg.get("run", {})
    env_cfg = cfg.get("env", {})
    arg_cfg, deprecation_warnings = _normalize_arg_config(cfg.get("args", {}))
    for alias in deprecated_override_aliases:
        canonical = ALIAS_KEYS[alias]
        message = f"Config key '{alias}' is deprecated and will be removed in a future cleanup; use '{canonical}' instead."
        if message not in deprecation_warnings:
            deprecation_warnings.append(message)

    python_cmd = run_cfg.get("python", sys.executable)
    if run_cfg.get("python") == "python3" and Path(sys.executable).name != "python3":
        print(
            "[Warning] Config requested 'python3', which may bypass the active virtual environment; "
            "prefer omitting run.python or setting it explicitly to the desired interpreter.",
            file=sys.stderr,
        )
    entry = run_cfg.get("entry", "main_multi_GPU.py")
    entry_path = (REPO_ROOT / entry).resolve()

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

    for message in deprecation_warnings:
        print(f"[Deprecated] {message}", file=sys.stderr)

    for r_start in r_starts:
        cmd = _build_command(python_cmd, entry_path, arg_cfg, r_start=r_start)
        print("[Run]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(REPO_ROOT), env=run_env, check=True)


if __name__ == "__main__":
    main()
