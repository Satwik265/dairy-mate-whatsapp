"""
run_pipeline.py

Runs the entire teat classifier pipeline end-to-end with one command:
    prepare_dataset.py -> train_teat_model.py -> evaluate.py -> gradcam_visualize.py

Stops immediately if any step fails, so you never train on a bad dataset or
evaluate a model that didn't actually finish training.

Run:
    python scripts/run_pipeline.py

Optional flags (passed through to train_teat_model.py):
    python scripts/run_pipeline.py --epochs_head 20 --epochs_finetune 15

Skip the (slow) Grad-CAM step if you just want a quick train+eval cycle:
    python scripts/run_pipeline.py --skip_gradcam
"""

import sys
import subprocess
import argparse
import time

STEPS = ["prepare_dataset.py", "train_teat_model.py", "evaluate.py", "gradcam_visualize.py"]


def run_step(script_name, extra_args=None):
    extra_args = extra_args or []
    cmd = [sys.executable, f"scripts/{script_name}"] + extra_args
    print(f"\n{'=' * 70}")
    print(f"  RUNNING: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n{'!' * 70}")
        print(f"  FAILED: {script_name} (exit code {result.returncode}, after {elapsed:.0f}s)")
        print(f"  Pipeline stopped. Fix the error above, then re-run.")
        print(f"{'!' * 70}\n")
        sys.exit(result.returncode)

    print(f"\n  OK: {script_name} finished in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_head", type=int, default=None)
    parser.add_argument("--epochs_finetune", type=int, default=None)
    parser.add_argument("--skip_gradcam", action="store_true",
                         help="Skip the Grad-CAM visualization step (useful for a quick train+eval cycle).")
    args = parser.parse_args()

    overall_start = time.time()

    run_step("prepare_dataset.py")

    train_args = []
    if args.epochs_head is not None:
        train_args += ["--epochs_head", str(args.epochs_head)]
    if args.epochs_finetune is not None:
        train_args += ["--epochs_finetune", str(args.epochs_finetune)]
    run_step("train_teat_model.py", train_args)

    run_step("evaluate.py")

    if not args.skip_gradcam:
        run_step("gradcam_visualize.py")
    else:
        print("\nSkipping gradcam_visualize.py (--skip_gradcam was set)")

    total_elapsed = time.time() - overall_start
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE in {total_elapsed / 60:.1f} minutes")
    print(f"  Model:      models/teat_classifier.keras")
    print(f"  Metrics:    reports/teat_evaluation_report.json")
    print(f"  Confusion:  reports/teat_confusion_matrix.png")
    if not args.skip_gradcam:
        print(f"  Grad-CAM:   reports/gradcam/gradcam_summary_grid.png")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
