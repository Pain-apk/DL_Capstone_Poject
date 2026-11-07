"""
run_pilot_pipeline.py - Orchestrates the automated pilot model training workflow:
1. Runs 50-step smoke test.
2. Checks the 3 continuation gates:
   - Peak VRAM < 7.2 GB
   - Loss decreasing
   - No errors
3. If gates pass, prints estimated training time and launches full 2-epoch training.
4. Checks eval/cases.json and reports status.
"""

import os
import sys
import json
import time
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMOKE_METRICS = os.path.join(PROJECT_ROOT, "outputs", "pilot_unverified", "smoke_test_metrics.json")
EVAL_CASES = os.path.join(PROJECT_ROOT, "eval", "cases.json")
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")

def run_pipeline():
    print("================================================================================")
    print("STEP 1: LAUNCHING 50-STEP SMOKE TEST")
    print("================================================================================\n")

    cmd_smoke = [VENV_PYTHON, os.path.join(PROJECT_ROOT, "src", "train.py"), "--smoke-test"]
    p_smoke = subprocess.run(cmd_smoke, cwd=PROJECT_ROOT)

    if p_smoke.returncode != 0:
        print(f"\n[GATE FAILURE] Smoke test exited with error code {p_smoke.returncode}!")
        return False

    if not os.path.exists(SMOKE_METRICS):
        print(f"\n[GATE FAILURE] Smoke test metrics file not found at {SMOKE_METRICS}!")
        return False

    with open(SMOKE_METRICS, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("\n--- Smoke Test Gate Verification ---")
    vram_pass = metrics.get("vram_gate_passed", False)
    loss_pass = metrics.get("loss_decreasing_gate_passed", False)
    peak_vram = metrics.get("peak_vram_gb", 0.0)
    init_loss = metrics.get("initial_loss_avg", 0.0)
    final_loss = metrics.get("final_loss_avg", 0.0)
    sec_per_step = metrics.get("sec_per_step", 1.0)

    print(f"Condition 1 - Peak VRAM: {peak_vram:.2f} GB (Strictly < 7.2 GB limit) -> {'PASSED' if vram_pass else 'FAILED'}")
    print(f"Condition 2 - Loss Trend: {init_loss:.4f} -> {final_loss:.4f} -> {'PASSED' if loss_pass else 'FAILED'}")
    print(f"Condition 3 - Training Errors: None -> PASSED")

    overall_pass = vram_pass and loss_pass
    if not overall_pass:
        print("\n<IMPORTANT> AUTOMATIC GATE FAILED. Stopping full training run.")
        print("Logs and metrics preserved under outputs/pilot_unverified/.")
        return False

    print("\n<IMPORTANT> ALL 3 CONTINUATION GATES PASSED! Proceeding to Full 2-Epoch Training.")

    # Total steps calculation:
    # 1775 train examples, batch size 2, grad accum 4 = effective batch size 8
    # Steps per epoch = 1775 / 8 = 222 steps.
    # 2 epochs = ~444 steps.
    total_steps = 444
    est_total_sec = total_steps * sec_per_step
    est_min = est_total_sec / 60.0
    print(f"Estimated full training duration: ~{est_min:.1f} minutes ({total_steps} steps @ {sec_per_step:.2f} s/step)")

    print("\n================================================================================")
    print("STEP 2: LAUNCHING FULL 2-EPOCH PILOT MODEL TRAINING")
    print("================================================================================\n")

    cmd_full = [VENV_PYTHON, os.path.join(PROJECT_ROOT, "src", "train.py")]
    p_full = subprocess.run(cmd_full, cwd=PROJECT_ROOT)

    if p_full.returncode != 0:
        print(f"\n[TRAINING FAILURE] Full training exited with error code {p_full.returncode}!")
        return False

    print("\n================================================================================")
    print("STEP 3: EVALUATION CHECK")
    print("================================================================================\n")
    if os.path.exists(EVAL_CASES):
        print(f"Found {EVAL_CASES}. Running evaluation...")
        eval_script = os.path.join(PROJECT_ROOT, "eval", "run_eval.py")
        if os.path.exists(eval_script):
            subprocess.run([VENV_PYTHON, eval_script], cwd=PROJECT_ROOT)
        else:
            print("eval/run_eval.py not yet implemented.")
    else:
        print(f"Evaluation skipped: {EVAL_CASES} does not exist yet. Hand-written evaluation will be performed once cases.json is supplied.")

    return True

if __name__ == "__main__":
    run_pipeline()

