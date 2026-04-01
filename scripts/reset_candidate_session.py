import argparse
import json
import shutil
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reset generated exam-session artifacts for a fresh candidate run."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting anything.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    return parser.parse_args()


def resolve_in_project(project_root: Path, path_text: str) -> Path:
    p = Path(path_text)
    resolved = p if p.is_absolute() else (project_root / p)
    resolved = resolved.resolve()
    resolved.relative_to(project_root.resolve())
    return resolved


def collect_targets(project_root: Path, config: dict):
    targets = []

    recording_dir = resolve_in_project(project_root, config["video"]["recording_path"])
    log_dir = resolve_in_project(project_root, config["logging"]["log_path"])
    reports_root = resolve_in_project(project_root, config["global"]["output_path"])
    reports_generated = resolve_in_project(project_root, config["reporting"]["output_dir"])
    reports_images = reports_generated / "images"
    violation_captures = reports_root / "violation_captures"
    uploads_dir = project_root / "uploads"

    targets.extend(
        [
            ("dir_contents", recording_dir),
            ("dir_contents", log_dir),
            ("dir_contents", reports_generated),
            ("dir_contents", reports_images),
            ("dir_contents", violation_captures),
            ("dir_contents", uploads_dir),
            ("file_reset_json", reports_root / "violations.json"),
        ]
    )

    return targets


def remove_dir_contents(path: Path, dry_run: bool):
    if not path.exists():
        return
    for child in path.iterdir():
        if dry_run:
            print(f"[DRY RUN] Remove: {child}")
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def reset_json_file(path: Path, dry_run: bool):
    if dry_run:
        print(f"[DRY RUN] Reset JSON file to []: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([], indent=2), encoding="utf-8")


def ensure_dirs_exist(targets, dry_run: bool):
    if dry_run:
        return
    for kind, path in targets:
        if kind == "dir_contents":
            path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    targets = collect_targets(project_root, config)

    print("Candidate session cleanup targets:")
    for kind, path in targets:
        print(f"- {kind}: {path}")

    if not args.yes and not args.dry_run:
        confirm = input("Proceed with cleanup? Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Cleanup cancelled.")
            return

    for kind, path in targets:
        if kind == "dir_contents":
            remove_dir_contents(path, args.dry_run)
        elif kind == "file_reset_json":
            reset_json_file(path, args.dry_run)

    ensure_dirs_exist(targets, args.dry_run)
    print("Cleanup complete." if not args.dry_run else "Dry run complete.")


if __name__ == "__main__":
    main()
