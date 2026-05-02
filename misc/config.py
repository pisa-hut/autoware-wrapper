#!/usr/bin/env python3

"""
Patch Autoware YAML configs in one shot with backup and extensible rules.

Usage:
  sudo ./autoware_patch_yaml.py --apply
  sudo ./autoware_patch_yaml.py --dry-run

Notes:
- Uses PyYAML for parsing/dumping. Formatting/comments may not be preserved.
- Creates a timestamp backup before writing: <file>.bak.<YYYYmmdd-HHMMSS>
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import os
import shutil
import sys
from typing import Any

try:
    import yaml  # PyYAML
except ImportError:
    print("[ERROR] PyYAML not found. Install it with:", file=sys.stderr)
    print("  python3 -m pip install pyyaml", file=sys.stderr)
    sys.exit(2)

Yaml = Any


def ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def backup_file(path: str) -> str:
    bak = f"{path}.bak.{ts()}"
    shutil.copy2(path, bak)
    return bak


def load_yaml(path: str) -> Yaml:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: str, data: Yaml) -> None:
    # sort_keys=False 以免 key 被重排太兇；仍可能改動格式（PyYAML限制）
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def deep_iter_nodes(obj: Yaml):
    """Yield all dict nodes in a nested YAML structure."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from deep_iter_nodes(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from deep_iter_nodes(it)


def set_param_under_any_ros_parameters(
    root: Yaml,
    param_key: str,
    value: Any,
    ros_keys: tuple[str, ...] = ("ros__parameters", "ros__paramters"),
) -> int:
    """
    Find every dict node that contains ros__parameters (or ros__paramters typo),
    and set param_key under it.
    Returns number of modifications performed.
    """
    n = 0
    for node in deep_iter_nodes(root):
        if not isinstance(node, dict):
            continue
        for rk in ros_keys:
            if rk in node and isinstance(node[rk], dict):
                before = node[rk].get(param_key, None)
                if before != value:
                    node[rk][param_key] = value
                    n += 1
    return n


def set_nested_key(root: Yaml, path: list[str | int], value: Any) -> bool:
    """
    Set a nested key by exact path (dict keys / list indices).
    Returns True if changed, False if path not found.
    """
    cur = root
    for p in path[:-1]:
        if isinstance(p, int):
            if not isinstance(cur, list) or p < 0 or p >= len(cur):
                return False
            cur = cur[p]
        else:
            if not isinstance(cur, dict) or p not in cur:
                return False
            cur = cur[p]

    last = path[-1]
    if isinstance(last, int):
        if not isinstance(cur, list) or last < 0 or last >= len(cur):
            return False
        if cur[last] != value:
            cur[last] = value
        return True
    else:
        if not isinstance(cur, dict):
            return False
        if cur.get(last) != value:
            cur[last] = value
        return True


def patch_topics_yaml_module_planning_warn_rate(
    root: Yaml,
    new_value: float,
    *,
    module_name: str = "planning",
    target_topic: str = "/planning/trajectory",
) -> int:
    """
    Only patch entries that satisfy:
      item["module"] == "planning"
      item["args"]["topic"] == "/planning/trajectory"

    Set:
      item["args"]["warn_rate"] = new_value
    """
    modified = 0

    # topics.yaml is usually a list, but be defensive
    if isinstance(root, list):
        items = root
    elif isinstance(root, dict):
        for k in ("topics", "modules"):
            if isinstance(root.get(k), list):
                items = root[k]
                break
        else:
            return 0
    else:
        return 0

    for item in items:
        if not isinstance(item, dict):
            continue

        if item.get("module") != module_name:
            continue

        args = item.get("args")
        if not isinstance(args, dict):
            continue

        if args.get("topic") != target_topic:
            continue

        if args.get("warn_rate") != new_value:
            args["warn_rate"] = float(new_value)
            modified += 1

    return modified


# -----------------------------
# Extensible patch rules
# -----------------------------
RULES = [
    {
        "name": "planning common max_vel -> 40.17",
        "file": "/opt/autoware/share/autoware_launch/config/planning/scenario_planning/common/common.param.yaml",
        "type": "rosparam",
        "param": "max_vel",
        "value": 40.17,
    },
    {
        "name": "AEB use_imu_path -> false",
        "file": "/opt/autoware/share/autoware_launch/config/control/autoware_autonomous_emergency_braking/autonomous_emergency_braking.param.yaml",
        "type": "rosparam",
        "param": "use_imu_path",
        "value": False,
    },
    {
        "name": "component_state_monitor topics planning warn_rate -> 1.0",
        "file": "/opt/autoware/share/autoware_launch/config/system/component_state_monitor/topics.yaml",
        "type": "topics_planning_warn_rate",
        "value": 1.0,
    },
    {
        "name": "operation_mode_transition_manager enable_engage_on_driving -> true",
        "file": "/opt/autoware/share/autoware_launch/config/control/operation_mode_transition_manager/operation_mode_transition_manager.param.yaml",
        "type": "rosparam",
        "param": "enable_engage_on_driving",
        "value": True,
    },
    {
        "name": "occlusion_spot detection_method -> predicted_object",
        "file": "/opt/autoware/share/autoware_launch/config/planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/occlusion_spot.param.yaml",
        "type": "rosparam_nested",
        "base_param": "occlusion_spot",  # occlusion_spot:
        "nested_key": "detection_method",  #   detection_method:
        "value": "predicted_object",
    },
    {
        "name": "enable_correct_goal_pose -> true",
        "file": "/opt/autoware/share/autoware_launch/config/planning/mission_planning/mission_planner/mission_planner.param.yaml",
        "type": "rosparam",
        "param": "enable_correct_goal_pose",
        "value": True,
    },
    {
        "name": "check_footprint_inside_lanes -> false",
        "file": "/opt/autoware/share/autoware_launch/config/planning/mission_planning/mission_planner/mission_planner.param.yaml",
        "type": "rosparam",
        "param": "check_footprint_inside_lanes",
        "value": False,
    },
    {
        "name": "allow_reroute_in_autonomous_mode -> true",
        "file": "/opt/autoware/share/autoware_launch/config/planning/mission_planning/mission_planner/mission_planner.param.yaml",
        "type": "rosparam",
        "param": "allow_reroute_in_autonomous_mode",
        "value": True,
    },
]


def apply_rule(data: Yaml, rule: dict[str, Any]) -> int:
    rtype = rule["type"]

    if rtype == "rosparam":
        return set_param_under_any_ros_parameters(data, rule["param"], rule["value"])

    if rtype == "rosparam_nested":
        # want: /**/ros__parameters/<base_param>/<nested_key> = value
        # We'll set it for every ros__parameters dict that contains base_param dict
        count = 0
        for node in deep_iter_nodes(data):
            if not isinstance(node, dict):
                continue
            for rk in ("ros__parameters", "ros__paramters"):
                if rk in node and isinstance(node[rk], dict):
                    rp = node[rk]
                    base = rp.get(rule["base_param"])
                    if base is None or not isinstance(base, dict):
                        continue
                    before = base.get(rule["nested_key"], None)
                    if before != rule["value"]:
                        base[rule["nested_key"]] = rule["value"]
                        count += 1
        return count

    if rtype == "topics_planning_warn_rate":
        return patch_topics_yaml_module_planning_warn_rate(data, rule["value"])

    raise ValueError(f"Unknown rule type: {rtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually write changes (with backup).")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change but do not write.",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Apply only rules whose name contains this substring.",
    )
    args = ap.parse_args()

    if not args.apply and not args.dry_run:
        ap.error("Please specify --apply or --dry-run")

    selected = []
    for r in RULES:
        if args.only and args.only not in r["name"]:
            continue
        selected.append(r)

    if not selected:
        print("[WARN] No rules selected.")
        return 0

    any_failed = False
    for r in selected:
        path = r["file"]
        print(f"\n=== {r['name']} ===")
        print(f"File: {path}")

        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            any_failed = True
            continue

        try:
            data = load_yaml(path)
        except Exception as e:
            print(f"[ERROR] Failed to parse YAML: {e}")
            any_failed = True
            continue

        original = copy.deepcopy(data)
        try:
            changed_count = apply_rule(data, r)
        except Exception as e:
            print(f"[ERROR] Failed applying rule: {e}")
            any_failed = True
            continue

        if changed_count == 0:
            # Still check if structure changed (shouldn't)
            if data == original:
                print("[OK] No change needed.")
            else:
                print("[WARN] Structure changed but no counted modifications (unexpected).")
        else:
            print(f"[OK] Modified occurrences: {changed_count}")

        if args.dry_run:
            continue

        # Write only if changed
        if data != original:
            try:
                bak = backup_file(path)
                dump_yaml(path, data)
                print(f"[WRITE] Updated. Backup: {bak}")
            except Exception as e:
                print(f"[ERROR] Failed writing YAML: {e}")
                any_failed = True
        else:
            print("[SKIP] Nothing to write.")

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
