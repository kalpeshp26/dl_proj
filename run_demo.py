"""
DeepRetail — one-command demo launcher.

Starts the FastAPI backend and the Streamlit dashboard in the background,
waits for the backend to come up, then runs the pipeline in the foreground.
Ctrl+C (or 'q' in the OpenCV window) stops everything.

Usage:
    python run_demo.py                 # live webcam (needs trained models)
    python run_demo.py --mock          # scripted demo, no camera / models needed
    python run_demo.py --source demo.mp4 --anomaly
    python run_demo.py --reset         # restore inventory from data/inventory.csv first

Any extra arguments are passed straight to src/pipeline/run_pipeline.py.
"""

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import API_BASE_URL, API_HOST, API_PORT  # noqa: E402


def wait_for_backend(timeout: float = 30.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"{API_BASE_URL}/pipeline/status", timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-dashboard", action="store_true", help="Don't start Streamlit")
    p.add_argument("--reset", action="store_true", help="Reset inventory & carts before starting")
    args, pipeline_args = p.parse_known_args()

    py = sys.executable
    procs: list[subprocess.Popen] = []
    try:
        print(f"[Demo] Starting backend on {API_BASE_URL} ...")
        procs.append(subprocess.Popen(
            [py, "-m", "uvicorn", "src.backend.main:app",
             "--host", API_HOST, "--port", str(API_PORT), "--log-level", "warning"],
            cwd=ROOT))
        if not wait_for_backend():
            sys.exit("[Demo] Backend did not start — is port 8000 already in use?")
        print("[Demo] Backend is up.")

        if args.reset:
            req = urllib.request.Request(f"{API_BASE_URL}/admin/reset", method="POST")
            urllib.request.urlopen(req, timeout=5).close()
            print("[Demo] Inventory and carts reset.")

        if not args.no_dashboard:
            print("[Demo] Starting Streamlit Analytics Hub on http://localhost:8501 ...")
            procs.append(subprocess.Popen(
                [py, "-m", "streamlit", "run", "dashboard/app.py",
                 "--server.headless", "true"], cwd=ROOT))
            import webbrowser
            time.sleep(1.5)
            print("[Demo] Enterprise Kiosk Web App: http://localhost:8000")
            print("[Demo] Streamlit Analytics Hub:  http://localhost:8501")
            webbrowser.open("http://localhost:8000")

        print(f"[Demo] Starting pipeline {' '.join(pipeline_args)}".rstrip())
        subprocess.run([py, "-m", "src.pipeline.run_pipeline", *pipeline_args], cwd=ROOT)
    except KeyboardInterrupt:
        pass
    finally:
        print("[Demo] Shutting down ...")
        for proc in reversed(procs):
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
