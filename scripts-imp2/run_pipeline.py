"""
run_imp2_pipeline.py: Execute the complete scripts-imp2 pipeline end-to-end.

This script runs:
1. Data loading (1-2 min)
2. Baseline GAT training (5-15 min)
3. Quantum GAT training (5-15 min)
4. Model comparison (1 min)

Total time: ~15-35 minutes depending on hardware
"""

import subprocess
import sys
from pathlib import Path
import time

script_dir = Path(__file__).parent
scripts = [
    ("Data Loading", "01_load_data.py"),
    ("Baseline GAT Training", "02_train_baseline_gat.py"),
    ("Quantum GAT Training", "03_train_quantum_gat.py"),
    ("Model Comparison", "04_compare_metrics.py"),
]

print("=" * 70)
print("QUANTUM-ENHANCED GAT FRAUD DETECTION PIPELINE")
print("=" * 70)
print(f"Starting complete pipeline execution...\n")

start_time = time.time()
failed = []

for idx, (name, script) in enumerate(scripts, 1):
    script_path = script_dir / script
    
    if not script_path.exists():
        print(f"\n✗ STEP {idx}/{len(scripts)}: {name}")
        print(f"  Script not found: {script_path}")
        failed.append(name)
        continue
    
    print(f"\n{'=' * 70}")
    print(f"STEP {idx}/{len(scripts)}: {name}")
    print(f"{'=' * 70}")
    print(f"Running: {script}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_dir.parent),
            check=True,
            capture_output=False,
        )
        print(f"\n✓ STEP {idx}/{len(scripts)}: {name} COMPLETED")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ STEP {idx}/{len(scripts)}: {name} FAILED")
        print(f"  Exit code: {e.returncode}")
        failed.append(name)
        # Continue to next step instead of aborting
    except KeyboardInterrupt:
        print(f"\n⚠ Pipeline interrupted by user")
        sys.exit(1)

total_time = time.time() - start_time

# Summary
print("\n" + "=" * 70)
print("PIPELINE EXECUTION SUMMARY")
print("=" * 70)

if not failed:
    print("\n✓ ALL STEPS COMPLETED SUCCESSFULLY!\n")
    print("Generated artifacts:")
    print("  • artifacts/elliptic_graph.pt")
    print("  • artifacts/baseline_gat_best.pt")
    print("  • artifacts/baseline_gat_metrics.json")
    print("  • artifacts/baseline_gat_training_curves.png")
    print("  • artifacts/quantum_gat_best.pt")
    print("  • artifacts/quantum_gat_metrics.json")
    print("  • artifacts/quantum_gat_training_curves.png")
    print("  • artifacts/model_comparison.png")
    print("  • artifacts/model_comparison_report.json")
    print(f"\nTotal execution time: {total_time / 60:.1f} minutes")
else:
    print(f"\n⚠ {len(failed)} step(s) failed:\n")
    for name in failed:
        print(f"  • {name}")
    print(f"\nCompleted steps: {len(scripts) - len(failed)}/{len(scripts)}")
    print(f"Execution time: {total_time / 60:.1f} minutes")

print("\n" + "=" * 70)
