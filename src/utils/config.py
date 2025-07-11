# general
import subprocess
from pathlib import Path


def get_repo_root():
    # Run 'git rev-parse --show-toplevel' command to get the root directory of the Git repository
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if git_root.returncode == 0:
        return Path(git_root.stdout.strip())
    else:
        raise RuntimeError("Unable to determine Git repository root directory.")


"""
Defines globals used throughout the codebase.
"""

###############################################################################
# Folder structure naming system
###############################################################################

# REPO DIRECTORIES
repo_dir = get_repo_root()
module_dir = repo_dir / "bleachbase"
data_dir = repo_dir / "data"
