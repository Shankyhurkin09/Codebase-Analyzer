"""Start the Codebase Analyzer app. Uses only pre-downloaded models in models/hf/."""

import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_PROJECT_ROOT)

if __name__ == "__main__":
    sys.exit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app_ui.py", "--server.headless", "true"],
        ).returncode
    )
