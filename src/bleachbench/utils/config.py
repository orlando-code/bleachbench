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

boundary_dir = data_dir / "boundaries"
sst_dir = data_dir / "sst"
crw_sst_dir = sst_dir / "CRW_SST"
bleaching_dir = data_dir / "bleaching"
unep_gdcr_dir = data_dir / "UNEP_WCMC"

###############################################################################
# Case study areas
###############################################################################

GBR_LONS = (140, 150)
GBR_LATS = (-20, -10)