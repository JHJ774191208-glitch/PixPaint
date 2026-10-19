#!/usr/bin/env python3
"""Evaluate paired inpainting results on complete RGB images.

The evaluator reports SSIM, L1 (%), LPIPS, and PSNR.  Metrics are computed
between each complete prediction and its complete ground-truth image; masks
are intentionally not used by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PREDICTION_SUFFIXES = ("_restored", "_prediction", "_pred", "_output")
SUPPORTED_METRICS = ("ssim", "l1", "lpips", "psnr")


@dataclass(frozen=True)
class ImagePair:
    key: str
    ground_truth: Path
    prediction: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required when --config is used. Install requirements.txt."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def _build_parser(config_defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute full-image SSIM, L1 (%), LPIPS, and PSNR for paired "
            "ground-truth and inpainting-result folders."
        )
    )
    parser.add_argument("--config", type=Path, help="Optional YAML configuration file.")
    parser.add_argument("--ground-truth", type=Path, required=False)
    parser.add_argument("--predictions", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_METRICS,
        default=list(SUPPORTED_METRICS),
    )
    parser.add_argument("--lpips-backbone", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--lpips-version", choices=("0.0", "0.1"), default="0.1")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search both input directories recursively.",
    )
    parser.add_argument(
        "--strict-pairing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any prediction or ground-truth image is unpaired.",
    )
    parser.add_argument(
        "--allow-lossy-ground-truth",
        action="store_true",
        help="Allow JPEG ground truth. PNG is recommended for quantitative evaluation.",
    )
    if config_defaults:
        unknown = sorted(set(config_defaults) - {action.dest for action in parser._actions})
        if unknown:
            parser.error("Unknown configuration keys: " + ", ".join(unknown))
        parser.set_defaults(**config_defaults)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    defaults = _load_yaml(pre_args.config) if pre_args.config else None
    args = _build_parser(defaults).parse_args(argv)
    if args.ground_truth is None or args.predictions is None:
        raise SystemExit("--ground-truth and --predictions are required (CLI or YAML).")
    args.metrics = list(dict.fromkeys(args.metrics))
    return args


def discover_images(root: Path, recursive: bool) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().lower(),
    )


def _normalised_stem(path: Path, prediction: bool) -> str:
    stem = path.stem.casefold()
    if prediction:
        for suffix in PREDICTION_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
    return stem


def _unique_index(paths: Sequence[Path], prediction: bool) -> tuple[dict[str, Path], set[str]]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in paths:
        key = _normalised_stem(path, prediction=prediction)
        if key in index:
            duplicates.add(key)
        else:
            index[key] = path
    for key in duplicates:
        index.pop(key, None)
    return index, duplicates


def pair_images(
    ground_truth_root: Path,
    prediction_root: Path,
    recursive: bool,
    strict: bool,
) -> tuple[list[ImagePair], list[str], list[str]]:
    ground_truth_files = discover_images(ground_truth_root, recursive)
    prediction_files = discover_images(prediction_root, recursive)
    if not ground_truth_files:
        raise RuntimeError(f"No supported images found in {ground_truth_root}")
    if not prediction_files:
        raise RuntimeError(f"No supported images found in {prediction_root}")

    gt_index, gt_duplicates = _unique_index(ground_truth_files, prediction=False)
    pred_index, pred_duplicates = _unique_index(prediction_files, prediction=True)
    if gt_duplicates or pred_duplicates:
        details = []
        if gt_duplicates:
            details.append("duplicate GT stems: " + ", ".join(sorted(gt_duplicates)))
        if pred_duplicates:
            details.append("duplicate prediction stems: " + ", ".join(sorted(pred_duplicates)))
        raise RuntimeError("Ambiguous filename pairing; " + "; ".join(details))

    common = sorted(set(gt_index) & set(pred_index))
    missing_predictions = sorted(set(gt_index) - set(pred_index))
    missing_ground_truth = sorted(set(pred_index) - set(gt_index))
    if strict and (missing_predictions or missing_ground_truth):
        message = [
            f"Strict pairing failed: {len(missing_predictions)} GT image(s) lack predictions; "
            f"{len(missing_ground_truth)} prediction(s) lack GT."
        ]
        if missing_predictions:
            message.append("Missing predictions (first 10): " + ", ".join(missing_predictions[:10]))
        if missing_ground_truth:
            message.append("Missing GT (first 10): " + ", ".join(missing_ground_truth[:10]))
        raise RuntimeError("\n".join(message))
    if not common:
        raise RuntimeError("No image pairs were found. Pair images using matching filename stems.")

    pairs = [
        ImagePair(key=key, ground_truth=gt_index[key], prediction=pred_index[key])
        for key in common
    ]
    return pairs, missing_predictions, missing_ground_truth


def load_rgb_float(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return array


def calculate_reference_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    metrics: Sequence[str],
) -> dict[str, float]:
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Image dimensions differ: GT {ground_truth.shape}, prediction {prediction.shape}"
        )
    result: dict[str, float] = {}
    if "ssim" in metrics:
        result["ssim"] = float(
            structural_similarity(ground_truth, prediction, channel_axis=2, data_range=1.0)
        )
    if "l1" in metrics:
        result["l1_percent"] = float(np.mean(np.abs(ground_truth - prediction)) * 100.0)
    if "psnr" in metrics:
        result["psnr_db"] = float(
            peak_signal_noise_ratio(ground_truth, prediction, data_range=1.0)
        )
    return result


class LPIPSEvaluator:
    def __init__(self, backbone: str, version: str, device: str) -> None:
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS requires torch, torchvision, and lpips. Install the documented dependencies."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
        self._torch = torch
        self._device = torch.device(device)
        self._model = lpips.LPIPS(net=backbone, version=version).to(self._device).eval()

    def __call__(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        torch = self._torch
        gt = torch.from_numpy(np.ascontiguousarray(ground_truth.transpose(2, 0, 1))).unsqueeze(0)
        pred = torch.from_numpy(np.ascontiguousarray(prediction.transpose(2, 0, 1))).unsqueeze(0)
        gt = gt.to(self._device, dtype=torch.float32).mul_(2.0).sub_(1.0)
        pred = pred.to(self._device, dtype=torch.float32).mul_(2.0).sub_(1.0)
        with torch.inference_mode():
            return float(self._model(gt, pred).item())


def _safe_mean(values: Sequence[float]) -> float:
    if any(math.isinf(value) for value in values):
        return math.inf
    return float(statistics.fmean(values))


def _safe_std(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.pstdev(finite)) if len(finite) > 1 else 0.0


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gt_root = args.ground_truth.resolve()
    pred_root = args.predictions.resolve()
    if not args.allow_lossy_ground_truth:
        lossy = [p for p in discover_images(gt_root, args.recursive) if p.suffix.lower() in {".jpg", ".jpeg"}]
        if lossy:
            raise RuntimeError(
                "JPEG ground truth detected. Use lossless PNG for reproducible metrics, or pass "
                "--allow-lossy-ground-truth if this is intentional. First file: " + str(lossy[0])
            )

    pairs, missing_predictions, missing_ground_truth = pair_images(
        gt_root, pred_root, args.recursive, args.strict_pairing
    )
    lpips_evaluator = None
    if "lpips" in args.metrics:
        lpips_evaluator = LPIPSEvaluator(args.lpips_backbone, args.lpips_version, args.device)

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        gt = load_rgb_float(pair.ground_truth)
        pred = load_rgb_float(pair.prediction)
        row: dict[str, Any] = {
            "image": pair.key,
            "ground_truth": pair.ground_truth.relative_to(gt_root).as_posix(),
            "prediction": pair.prediction.relative_to(pred_root).as_posix(),
            "width": int(gt.shape[1]),
            "height": int(gt.shape[0]),
        }
        row.update(calculate_reference_metrics(gt, pred, args.metrics))
        if lpips_evaluator is not None:
            row["lpips"] = lpips_evaluator(gt, pred)
        rows.append(row)
        values = ", ".join(
            f"{key}={value:.6f}" for key, value in row.items() if isinstance(value, float)
        )
        print(f"[{index:04d}/{len(pairs):04d}] {pair.key}: {values}", flush=True)

    metric_columns = [
        key for key in ("ssim", "l1_percent", "lpips", "psnr_db") if key in rows[0]
    ]
    summary_metrics = {
        key: {
            "mean": _safe_mean([float(row[key]) for row in rows]),
            "std_population": _safe_std([float(row[key]) for row in rows]),
        }
        for key in metric_columns
    }
    return {
        "rows": rows,
        "summary": {
            "evaluation_region": "full_image",
            "ground_truth_dir": str(gt_root),
            "predictions_dir": str(pred_root),
            "paired_images": len(rows),
            "missing_predictions": missing_predictions,
            "missing_ground_truth": missing_ground_truth,
            "strict_pairing": bool(args.strict_pairing),
            "metrics": summary_metrics,
            "lpips_backbone": args.lpips_backbone if "lpips" in args.metrics else None,
            "lpips_version": args.lpips_version if "lpips" in args.metrics else None,
            "device": args.device if "lpips" in args.metrics else None,
        },
    }


def _json_number(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_number(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_number(item) for item in value]
    return value


def save_results(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    per_image_path = output_dir / "metrics_per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_number(result["summary"]), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\nFull-image evaluation summary")
    print(f"Paired images: {result['summary']['paired_images']}")
    for metric, values in result["summary"]["metrics"].items():
        print(f"{metric}: mean={values['mean']:.6f}, std={values['std_population']:.6f}")
    print(f"Per-image CSV: {per_image_path.resolve()}")
    print(f"Summary JSON:  {summary_path.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = evaluate(args)
        save_results(result, args.output_dir)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
