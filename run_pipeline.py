"""Run the full EcoScan AI pipeline.
Images must be collected first (takes ~1 h because of API rate limits):
    python src/collect_online.py
then this script runs:
TrashNet external test download -> clean -> label check -> near-duplicate grouping -> split (leakage check) -> train (tree, CNN, hybrid)
-> unknown-image threshold -> evaluate.

    python run_pipeline.py            # everything
    python run_pipeline.py --from train_cnn
"""
import subprocess
import sys
from pathlib import Path

STEPS = ["prepare_data", "clean", "label_check", "near_duplicates", "split", "train_tree", "train_cnn", "train_hybrid", "ood", "evaluate"]
SRC = Path(__file__).resolve().parent / "src"

start = STEPS.index(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
for step in STEPS[start:]:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, f"{step}.py"], cwd=SRC, check=True)
