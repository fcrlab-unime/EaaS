"""
Local end-to-end pipeline test: training → ONNX export → quantization.
No containers, no Kafka, no MinIO — just local files.

Usage:
    python test_pipeline.py --csv df_clean.csv [options]

Options:
    --csv           Path to dataset CSV (required)
    --label         Label column name (default: last column)
    --model-script  Path to model .py file (default: CCN1D.py)
    --class-name    Model class name (default: CCN1D)
    --model-params  JSON string with model constructor kwargs
                    (default: {"input_channels":<n_features>, "hidden_channels":16, "num_layers":2})
    --patience      Early-stopping patience (default: 10)
    --max-epochs    Max training epochs (default: 500)
    --lr            Learning rate (default: 1e-3)
    --output-dir    Directory for all output files (default: ./pipeline_test_output)
    --seed          Global seed (default: 42)
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_pipeline")

# ── make training/ and quantization/ importable without installing them ───────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "quantization"))

from training.train import train, apply_reproducibility  # noqa: E402  (training/train.py)
from quantization.quant import PipelineConfig  # noqa: E402  (quantization/quant.py)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model_class(script_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("_user_model", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def export_onnx(model, input_channels: int, onnx_path: str, seed: int):
    torch.manual_seed(seed)
    dummy = torch.randn(1, input_channels, 1)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        do_constant_folding=True,
    )
    import onnx as _onnx
    m = _onnx.load(onnx_path)
    _onnx.save_model(m, onnx_path, save_as_external_data=False)
    log.info("ONNX exported (inline): %s", onnx_path)


def parse_args():
    p = argparse.ArgumentParser(description="Local pipeline test: train → ONNX → quantize")
    p.add_argument("--csv", required=True, help="Path to dataset CSV")
    p.add_argument("--label", default=None, help="Label column (default: last column)")
    p.add_argument("--model-script", default=None,
                   help="Path to model .py (default: CCN1D.py in project root)")
    p.add_argument("--class-name", default="CCN1D", help="Model class name (default: CCN1D)")
    p.add_argument("--model-params", default=None,
                   help='JSON model constructor kwargs, e.g. \'{"input_channels":10,"hidden_channels":16,"num_layers":2}\'')
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", default="pipeline_test_output")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.environ.setdefault("TRAINING_SEED", str(args.seed))
    os.environ.setdefault("QUANT_SEED", str(args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found: {csv_path}")

    # Infer n_features to build default model params
    df_head = pd.read_csv(csv_path, nrows=1)
    label_col = args.label or df_head.columns[-1]
    n_features = len(df_head.columns) - 1  # all cols except label

    # Model script
    model_script = args.model_script or str(ROOT / "CCN1D.py")
    if not Path(model_script).exists():
        sys.exit(f"ERROR: model script not found: {model_script}")

    # Model params
    if args.model_params:
        model_params = json.loads(args.model_params)
    else:
        model_params = {
            "input_channels": n_features,
            "hidden_channels": 512,
            "num_layers": 4,
        }
    log.info("Model: %s  params: %s", args.class_name, model_params)

    # Seed BEFORE model instantiation so weight init is deterministic.
    # train() calls apply_reproducibility() again internally, which is fine:
    # it resets the RNG to the same state before the training loop.
    apply_reproducibility()

    # ── 1. TRAINING ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1 — Training")
    log.info("=" * 60)

    ModelClass = load_model_class(model_script, args.class_name)
    model = ModelClass(**model_params)

    model, train_metrics, split_dfs = train(
        model,
        str(csv_path),
        patience=args.patience,
        max_epochs=args.max_epochs,
        label_column=label_col,
        lr=args.lr,
    )

    log.info(
        "Training done → MAE: %.4f | RMSE: %.4f | R²: %.4f",
        train_metrics["mae"], train_metrics["rmse"], train_metrics["r2"],
    )

    # Save split CSVs
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    split_paths = {}
    for name, df in split_dfs.items():
        p = splits_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        split_paths[name] = str(p)
        log.info("Split '%s' saved: %s (%d rows)", name, p, len(df))

    # Save .pth
    pth_path = str(out_dir / f"{args.class_name}_trained.pth")
    torch.save(model.state_dict(), pth_path)
    log.info(".pth saved: %s", pth_path)

    # ── 2. ONNX EXPORT ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 2 — ONNX export")
    log.info("=" * 60)

    onnx_path = str(out_dir / f"{args.class_name}_final.onnx")
    export_onnx(model, model_params["input_channels"], onnx_path, args.seed)

    # ── 3. QUANTIZATION ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 3 — Quantization")
    log.info("=" * 60)

    cfg = PipelineConfig(
        model_path=onnx_path,
        train_csv=split_paths["train"],
        label=label_col,
    )

    test_csv_for_calib = split_paths.get("test")
    quant_model = cfg.quant(calib_csv=split_paths["train"])
    log.info("Quantization complete")

    # Save quantized model
    import aidge_core.aidge_save as _aidge_save
    quant_model_path = str(out_dir / "model_quant")
    _aidge_save.save_model(quant_model, quant_model_path)
    log.info("Quantized model saved: %s.aidge", quant_model_path)

    # ── 4. POST-QUANT METRICS ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 4 — Post-quantization metrics")
    log.info("=" * 60)

    if test_csv_for_calib:
        try:
            quant_metrics = cfg.compute_metrics(quant_model, test_csv_for_calib)
            log.info(
                "Post-quant → MAE: %.4f | RMSE: %.4f | R²: %.4f",
                quant_metrics["mae"], quant_metrics["rmse"], quant_metrics["r2"],
            )
        except Exception as exc:
            log.warning("Post-quant metrics failed: %s", exc)
            quant_metrics = None
    else:
        log.warning("No test split available — skipping post-quant metrics")
        quant_metrics = None

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    log.info("Float model — MAE: %.4f | RMSE: %.4f | R²: %.4f",
             train_metrics["mae"], train_metrics["rmse"], train_metrics["r2"])
    if quant_metrics:
        log.info("Quant model — MAE: %.4f | RMSE: %.4f | R²: %.4f",
                 quant_metrics["mae"], quant_metrics["rmse"], quant_metrics["r2"])
        mae_delta = quant_metrics["mae"] - train_metrics["mae"]
        r2_delta  = quant_metrics["r2"]  - train_metrics["r2"]
        log.info("Delta       — ΔMAE: %+.4f | ΔR²: %+.4f", mae_delta, r2_delta)
    log.info("Output dir: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
