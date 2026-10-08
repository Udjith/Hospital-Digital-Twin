from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def main():
    dashboard = PROJECT_ROOT / "src" / "dashboard.py"

    if not dashboard.exists():
        raise FileNotFoundError(
            f"Dashboard not found at: {dashboard}"
        )

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")


if __name__ == "__main__":
    main()