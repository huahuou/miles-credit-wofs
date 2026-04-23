#!/usr/bin/env python3
"""Detect unreadable/corrupted zarr datasets from config globs.

Default behavior is detection-only. The script writes a bad-file report so you
can decide whether to quarantine or remove files later.

Example:
  python scripts/detect_corrupt_zarr.py \
    --config config/wofs_credit_wrf_da_increment.yml

Optional quarantine:
  python scripts/detect_corrupt_zarr.py \
    --config config/wofs_credit_wrf_da_increment.yml \
    --move-bad-dir /scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/quarantine_zarr

python scripts/detect_corrupt_zarr.py --config config/wofs_credit_wrf_da_increment.yml --split all --timeout-sec 20 --move-bad-dir /scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/quarantine_zarr
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import time
import warnings
from glob import glob
from pathlib import Path

import yaml

from credit.data import filter_ds, get_forward_data


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _extract_case_date(file_path: str) -> str | None:
    match = re.search(r"wofs_(\d{8})_\d{4}_mem\d+\.zarr(?:\.zip)?$", Path(file_path).name)
    return match.group(1) if match else None


def _select_files(
    glob_pattern: str,
    years_range: list[int] | list[str] | None,
    date_range: list[str] | tuple[str, str] | None,
) -> list[str]:
    files = sorted(glob(glob_pattern))

    if date_range is not None:
        start_date, end_date = str(date_range[0]), str(date_range[1])
        selected = []
        for file_path in files:
            case_date = _extract_case_date(file_path)
            if case_date is not None and start_date <= case_date <= end_date:
                selected.append(file_path)
        return selected

    if years_range is None:
        return files

    if len(years_range) == 2:
        start_year, end_year = int(years_range[0]), int(years_range[1])
        years = [str(year) for year in range(start_year, end_year + 1)]
    else:
        years = [str(year) for year in years_range]

    return [file_path for file_path in files if any(year in file_path for year in years)]


def _build_scan_entries(conf: dict, split: str, include_dynamic: bool):
    data_conf = conf["data"]

    upper_vars = _dedupe_keep_order(
        list(data_conf.get("variables", []))
        + list(data_conf.get("context_upper_air_variables", []))
        + list(data_conf.get("observation_variables", []))
    )
    dyn_vars = list(data_conf.get("dynamic_forcing_variables", []))

    split_names = [split] if split in {"train", "valid"} else ["train", "valid"]
    entries = []
    metadata = {}

    for split_name in split_names:
        years_key = "train_years" if split_name == "train" else "valid_years"
        date_key = "train_date_range" if split_name == "train" else "valid_date_range"
        years_range = data_conf.get(years_key)
        date_range = data_conf.get(date_key)

        upper_pattern = data_conf["save_loc"]
        upper_files = _select_files(upper_pattern, years_range, date_range)
        if len(upper_files) == 0:
            raise ValueError(
                f"No files found for split={split_name} using pattern={upper_pattern} with date_range={date_range}"
            )

        dyn_files = None
        if include_dynamic and dyn_vars and data_conf.get("save_loc_dynamic_forcing"):
            dyn_pattern = data_conf["save_loc_dynamic_forcing"]
            dyn_files = _select_files(dyn_pattern, years_range, date_range)
            if len(dyn_files) == 0:
                raise ValueError(
                    f"No dynamic forcing files found for split={split_name} using pattern={dyn_pattern} with date_range={date_range}"
                )
            if len(dyn_files) != len(upper_files):
                raise ValueError(
                    "Mismatch between upper-air and dynamic forcing file counts "
                    f"for split={split_name}: upper={len(upper_files)} dyn={len(dyn_files)}"
                )

        metadata[split_name] = {
            "upper_count": len(upper_files),
            "dynamic_count": len(dyn_files) if dyn_files is not None else 0,
            "years_range": years_range,
            "date_range": date_range,
        }

        for idx, upper_path in enumerate(upper_files):
            entries.append(
                {
                    "split": split_name,
                    "upper_path": upper_path,
                    "dyn_path": dyn_files[idx] if dyn_files is not None else None,
                }
            )

    # Deduplicate in case split=all has overlaps.
    dedup = []
    seen = set()
    for entry in entries:
        key = (entry["upper_path"], entry["dyn_path"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(entry)

    return dedup, upper_vars, dyn_vars, metadata


def _open_filtered_with_retries(path: str, required_vars: list[str], zarr_time_chunk: int, max_open_retries: int):
    chunks = {"time": zarr_time_chunk} if zarr_time_chunk > 0 else None
    last_exc = None

    for _ in range(max(1, max_open_retries)):
        ds = None
        try:
            ds = get_forward_data(path, zarr_chunks=chunks)
            if required_vars:
                ds = filter_ds(ds, required_vars)
            return ds
        except Exception as exc:
            last_exc = exc
            try:
                if ds is not None:
                    ds.close()
            except Exception:
                pass

    raise last_exc


def _touch_first_timestep(ds):
    if "time" in ds.dims and int(ds.sizes.get("time", 0)) > 0:
        ds.isel(time=slice(0, 1)).load()
    else:
        ds.load()


def _extract_suspicious_zarr_warnings(caught_warnings):
    suspicious = []
    for w in caught_warnings:
        msg = str(w.message)
        if (
            "not recognized as a component of a Zarr hierarchy" in msg
            or "There is no item named" in msg
        ):
            suspicious.append(f"{type(w.message).__name__}: {msg}")
    return suspicious


def _probe_worker(
    entry: dict,
    upper_vars: list[str],
    dyn_vars: list[str],
    zarr_time_chunk: int,
    max_open_retries: int,
    validate_read: bool,
    fail_on_zarr_hierarchy_warning: bool,
    queue,
):
    upper_ds = None
    dyn_ds = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            upper_ds = _open_filtered_with_retries(entry["upper_path"], upper_vars, zarr_time_chunk, max_open_retries)
            n_time = int(upper_ds["time"].size)

            if validate_read:
                _touch_first_timestep(upper_ds)

            if entry.get("dyn_path") is not None and dyn_vars:
                dyn_ds = _open_filtered_with_retries(entry["dyn_path"], dyn_vars, zarr_time_chunk, max_open_retries)
                dyn_n_time = int(dyn_ds["time"].size)
                n_time = min(n_time, dyn_n_time)
                if validate_read:
                    _touch_first_timestep(dyn_ds)

            suspicious = _extract_suspicious_zarr_warnings(caught)
            if fail_on_zarr_hierarchy_warning and suspicious:
                queue.put(
                    {
                        "status": "bad",
                        "n_time": None,
                        "error": f"ZarrHierarchyWarning: {suspicious[0]}",
                        "warning_count": len(caught),
                        "suspicious_warning_count": len(suspicious),
                    }
                )
                return

            queue.put(
                {
                    "status": "ok",
                    "n_time": n_time,
                    "error": "",
                    "warning_count": len(caught),
                    "suspicious_warning_count": len(suspicious),
                }
            )
    except Exception as exc:
        queue.put(
            {
                "status": "bad",
                "n_time": None,
                "error": f"{type(exc).__name__}: {exc}",
                "warning_count": 0,
                "suspicious_warning_count": 0,
            }
        )
    finally:
        try:
            if upper_ds is not None:
                upper_ds.close()
        except Exception:
            pass
        try:
            if dyn_ds is not None:
                dyn_ds.close()
        except Exception:
            pass


def _probe_with_timeout(
    entry: dict,
    upper_vars: list[str],
    dyn_vars: list[str],
    timeout_sec: float,
    zarr_time_chunk: int,
    max_open_retries: int,
    validate_read: bool,
    fail_on_zarr_hierarchy_warning: bool,
):
    # Separate process prevents one stuck open/read from hanging the full scan.
    ctx = mp.get_context("fork")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_probe_worker,
        args=(
            entry,
            upper_vars,
            dyn_vars,
            zarr_time_chunk,
            max_open_retries,
            validate_read,
            fail_on_zarr_hierarchy_warning,
            queue,
        ),
        daemon=True,
    )

    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        return {
            "status": "bad",
            "n_time": None,
            "error": f"TimeoutError: probe exceeded {timeout_sec:.1f}s",
            "warning_count": 0,
            "suspicious_warning_count": 0,
        }

    if not queue.empty():
        return queue.get()

    if proc.exitcode == 0:
        return {
            "status": "bad",
            "n_time": None,
            "error": "UnknownError: probe exited without result",
            "warning_count": 0,
            "suspicious_warning_count": 0,
        }

    return {
        "status": "bad",
        "n_time": None,
        "error": f"WorkerExitError: exit code {proc.exitcode}",
        "warning_count": 0,
        "suspicious_warning_count": 0,
    }


def _safe_move(src_path: str, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    src = Path(src_path)
    dst = Path(dst_dir) / src.name
    if dst.exists():
        stem = dst.stem
        suffix = "".join(dst.suffixes)
        for idx in range(1, 100000):
            candidate = Path(dst_dir) / f"{stem}.dup{idx}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
    shutil.move(str(src), str(dst))
    return str(dst)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect corrupted/unreadable zarr or zarr.zip files mirroring WoFS training checks.")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config path (example: config/wofs_credit_wrf_da_increment.yml)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "all"],
        default="train",
        help="Which split selection logic to use from config (default: train).",
    )
    parser.add_argument(
        "--include-dynamic",
        dest="include_dynamic",
        action="store_true",
        help="Check dynamic forcing datasets too (default enabled).",
    )
    parser.add_argument(
        "--no-dynamic",
        dest="include_dynamic",
        action="store_false",
        help="Skip dynamic forcing checks.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=20.0,
        help="Per-file probe timeout in seconds.",
    )
    parser.add_argument(
        "--zarr-time-chunk",
        type=int,
        default=None,
        help="time chunk used in get_forward_data; defaults to config data.zarr_time_chunk.",
    )
    parser.add_argument(
        "--max-open-retries",
        type=int,
        default=0,
        help="Dataset open/filter retries; 0 means use config data.max_dataset_open_retries (or 3).",
    )
    parser.add_argument(
        "--validate-read",
        dest="validate_read",
        action="store_true",
        help="Load the first timestep after filtering (closer to training-time behavior; default enabled).",
    )
    parser.add_argument(
        "--metadata-only",
        dest="validate_read",
        action="store_false",
        help="Skip data load and only validate open/filter metadata.",
    )
    parser.add_argument(
        "--fail-on-zarr-hierarchy-warning",
        dest="fail_on_zarr_hierarchy_warning",
        action="store_true",
        help="Mark files bad when zarr hierarchy warnings are observed (default enabled).",
    )
    parser.add_argument(
        "--allow-zarr-hierarchy-warning",
        dest="fail_on_zarr_hierarchy_warning",
        action="store_false",
        help="Do not mark zarr hierarchy warnings as bad.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index into matched file list.")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to scan (0 means all).")
    parser.add_argument(
        "--report-prefix",
        default="zarr_corruption_scan",
        help="Prefix for output report files.",
    )
    parser.add_argument(
        "--move-bad-dir",
        default="",
        help="If set, move bad files into this directory after scan.",
    )
    parser.add_argument(
        "--delete-bad",
        action="store_true",
        help="If set, delete bad files after scan. Use with caution.",
    )
    parser.set_defaults(
        include_dynamic=True,
        validate_read=True,
        fail_on_zarr_hierarchy_warning=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be > 0")
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.max_files < 0:
        raise ValueError("--max-files must be >= 0")
    if args.max_open_retries < 0:
        raise ValueError("--max-open-retries must be >= 0")
    if args.move_bad_dir and args.delete_bad:
        raise ValueError("Use either --move-bad-dir or --delete-bad, not both.")

    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    zarr_time_chunk = (
        int(conf["data"].get("zarr_time_chunk", 2)) if args.zarr_time_chunk is None else int(args.zarr_time_chunk)
    )
    max_open_retries = (
        int(conf["data"].get("max_dataset_open_retries", 3)) if args.max_open_retries == 0 else int(args.max_open_retries)
    )

    entries, upper_vars, dyn_vars, split_meta = _build_scan_entries(conf, args.split, args.include_dynamic)
    total_found = len(entries)

    if args.start >= total_found:
        raise ValueError(f"--start={args.start} is beyond total matched files={total_found}")

    entries = entries[args.start :]
    if args.max_files > 0:
        entries = entries[: args.max_files]

    print("=== Zarr Corruption Scan ===")
    print(f"Config: {args.config}")
    print(f"Split mode: {args.split}")
    print(f"Include dynamic: {args.include_dynamic}")
    print(f"Required upper vars ({len(upper_vars)}): {upper_vars}")
    if args.include_dynamic:
        print(f"Required dynamic vars ({len(dyn_vars)}): {dyn_vars}")
    for split_name, meta in split_meta.items():
        print(
            f"  split={split_name}: upper={meta['upper_count']} dynamic={meta['dynamic_count']} "
            f"years={meta['years_range']} date_range={meta['date_range']}"
        )
    print(f"Unique file pairs found: {total_found}")
    print(f"File pairs to scan now: {len(entries)}")
    print(f"Per-file timeout: {args.timeout_sec:.1f}s")
    print(f"zarr_time_chunk: {zarr_time_chunk}")
    print(f"max_open_retries: {max_open_retries}")
    print(f"validate_read: {args.validate_read}")
    print(f"fail_on_zarr_hierarchy_warning: {args.fail_on_zarr_hierarchy_warning}")

    bad = []
    good = 0
    results = []
    warning_total = 0
    suspicious_warning_total = 0
    started = time.time()

    for idx, entry in enumerate(entries, start=1):
        result = _probe_with_timeout(
            entry,
            upper_vars,
            dyn_vars,
            args.timeout_sec,
            zarr_time_chunk,
            max_open_retries,
            args.validate_read,
            args.fail_on_zarr_hierarchy_warning,
        )
        status = result["status"]
        n_time = result["n_time"]
        error = result["error"]
        warning_count = int(result.get("warning_count", 0))
        suspicious_warning_count = int(result.get("suspicious_warning_count", 0))
        warning_total += warning_count
        suspicious_warning_total += suspicious_warning_count

        if status == "ok":
            good += 1
        else:
            bad.append(
                {
                    "split": entry["split"],
                    "upper_path": entry["upper_path"],
                    "dyn_path": entry.get("dyn_path"),
                    "error": error,
                }
            )

        results.append(
            {
                "split": entry["split"],
                "upper_path": entry["upper_path"],
                "dyn_path": entry.get("dyn_path"),
                "status": status,
                "n_time": n_time,
                "error": error,
                "warning_count": warning_count,
                "suspicious_warning_count": suspicious_warning_count,
            }
        )

        if idx % 50 == 0 or idx == len(entries):
            elapsed = time.time() - started
            rate = idx / elapsed if elapsed > 0 else 0.0
            print(
                f"Progress: {idx}/{len(entries)} scanned, good={good}, bad={len(bad)}, "
                f"warn={warning_total}, suspicious_warn={suspicious_warning_total}, rate={rate:.2f} files/s"
            )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.report_prefix}_{timestamp}"
    bad_txt = f"{prefix}.bad.txt"
    all_json = f"{prefix}.all.json"
    summary_json = f"{prefix}.summary.json"

    with open(bad_txt, "w", encoding="utf-8") as f:
        for item in bad:
            f.write(
                f"split={item['split']}\tupper={item['upper_path']}\tdyn={item['dyn_path']}\t{item['error']}\n"
            )

    with open(all_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    moved = []
    deleted = []
    action_errors = []

    action_paths = set()
    for item in bad:
        action_paths.add(item["upper_path"])
        dyn_path = item.get("dyn_path")
        if dyn_path:
            action_paths.add(dyn_path)

    if args.move_bad_dir:
        for path in sorted(action_paths):
            try:
                new_path = _safe_move(path, args.move_bad_dir)
                moved.append({"from": path, "to": new_path})
            except Exception as exc:
                action_errors.append({"path": path, "action": "move", "error": f"{type(exc).__name__}: {exc}"})

    if args.delete_bad:
        for path in sorted(action_paths):
            try:
                os.remove(path)
                deleted.append(path)
            except Exception as exc:
                action_errors.append({"path": path, "action": "delete", "error": f"{type(exc).__name__}: {exc}"})

    elapsed = time.time() - started
    summary = {
        "config": args.config,
        "split": args.split,
        "include_dynamic": args.include_dynamic,
        "required_upper_vars": upper_vars,
        "required_dynamic_vars": dyn_vars if args.include_dynamic else [],
        "total_found_unique_pairs": total_found,
        "scanned": len(entries),
        "good": good,
        "bad": len(bad),
        "warning_total": warning_total,
        "suspicious_warning_total": suspicious_warning_total,
        "timeout_sec": args.timeout_sec,
        "zarr_time_chunk": zarr_time_chunk,
        "max_open_retries": max_open_retries,
        "validate_read": args.validate_read,
        "fail_on_zarr_hierarchy_warning": args.fail_on_zarr_hierarchy_warning,
        "elapsed_sec": elapsed,
        "bad_report": bad_txt,
        "all_report": all_json,
        "moved_count": len(moved),
        "deleted_count": len(deleted),
        "action_errors": action_errors,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Scan Complete ===")
    print(f"Scanned: {len(entries)}")
    print(f"Good: {good}")
    print(f"Bad: {len(bad)}")
    print(f"Warnings observed: {warning_total}")
    print(f"Suspicious zarr hierarchy warnings: {suspicious_warning_total}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Bad list: {bad_txt}")
    print(f"All results: {all_json}")
    print(f"Summary: {summary_json}")

    if args.move_bad_dir:
        print(f"Moved bad files: {len(moved)} to {args.move_bad_dir}")
    if args.delete_bad:
        print(f"Deleted bad files: {len(deleted)}")
    if action_errors:
        print(f"Action errors: {len(action_errors)}")


if __name__ == "__main__":
    main()
