"""Run explicit CUDA memory tiers for the Stage 3 training graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from training.train_spherical_ba_recurrent_refiner import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sizes", nargs="+", default=["126x252", "252x504", "504x1008"])
    parser.add_argument("--output", default="outputs/stage3_cuda_smoke")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 3 CUDA smoke requires a visible CUDA GPU.")
    records = []
    for text in args.sizes:
        height, width = (int(value) for value in text.lower().split("x", maxsplit=1))
        config = load_config(args.config)
        config["train"].update(
            {
                "ddp": False,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_steps": 1,
                "diagnostics_interval": 10_000,
                "val_interval": 10_000,
                "save_interval": 1,
                "output_dir": str(Path(args.output) / text),
                "num_workers": 0,
            }
        )
        config["dataset"]["max_train_samples"] = 1
        config["refiner"]["profile_synchronize_cuda"] = True
        config.setdefault("Renderer", {})["profile_synchronize_cuda"] = True
        config["WeightsAndBiases"]["run_name"] = f"{config['WeightsAndBiases'].get('run_name', 'stage3')}_smoke_{text}"
        config["image"]["head_height"] = height
        config["image"]["head_width"] = width
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        status, error = "passed", None
        try:
            result = train(config)
        except Exception as exc:  # pragma: no cover - CUDA integration utility
            status, error, result = "failed", repr(exc), None
        elapsed = time.perf_counter() - start
        records.append(
            {
                "size": text,
                "status": status,
                "error": error,
                "elapsed_sec": elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "result": result,
            }
        )
        print(json.dumps(records[-1], indent=2), flush=True)
        if status != "passed":
            break
    output = Path(args.output) / "smoke_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
