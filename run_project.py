# ============================================================
# RUN_PROJECT.PY — Master Runner
# Runs all steps in order, then launches the dashboard
#
# HOW TO USE:
#   python run_project.py
# ============================================================

import subprocess
import sys
import os
import time

# ── Make sure we're in the right folder ───────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ── Colors for terminal output ────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

STEPS = [
    ("STEP 1 — Load & Explore Data",       "notebooks/STEP1_load_and_explore.py"),
    ("STEP 2 — Build SQL Database",        "notebooks/STEP2_sql_database.py"),
    ("STEP 3 — Download Weather Data",     "notebooks/STEP3_weather_download.py"),
    ("STEP 4 — Feature Engineering",       "notebooks/STEP4_feature_engineering.py"),
    ("STEP 5 — EDA + Maps",                "notebooks/STEP5_eda_and_maps.py"),
    ("STEP 6 — ML Models",                 "notebooks/STEP6_ml_models.py"),
    ("STEP 6B — TPE + Policy Simulations", "notebooks/STEP6B_tpe_simulations.py"),
]

def print_header():
    print(f"\n{BOLD}{'=' * 60}")
    print("   M PROJECT — FULL PIPELINE RUNNER")
    print(f"{'=' * 60}{RESET}\n")

def run_step(name, script):
    print(f"{BLUE}{BOLD}▶ Running {name}...{RESET}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False   # shows live output in terminal
    )

    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"{GREEN}✅ {name} completed in {elapsed:.1f}s{RESET}\n")
        return True
    else:
        print(f"{RED}❌ {name} FAILED (exit code {result.returncode}){RESET}")
        print(f"{YELLOW}   Fix the error above, then re-run this script.{RESET}\n")
        return False

def launch_dashboard():
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {GREEN}ALL STEPS COMPLETE! Launching Dashboard...{RESET}{BOLD}")
    print(f"{'=' * 60}{RESET}")
    print(f"\n{YELLOW}  Open your browser at: http://localhost:8501{RESET}\n")
    print(f"  Press Ctrl+C to stop the dashboard.\n")

    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    "notebooks/STEP7_dashboard.py"])

# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print_header()

    total_start = time.time()

    for name, script in STEPS:
        success = run_step(name, script)
        if not success:
            print(f"{RED}{BOLD}Pipeline stopped. Fix the error and re-run.{RESET}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"{GREEN}{BOLD}All {len(STEPS)} steps finished in {total_elapsed/60:.1f} minutes.{RESET}")

    launch_dashboard()
