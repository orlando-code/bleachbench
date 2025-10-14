from datetime import datetime
from pathlib import Path
import re

from bleachbench.utils import utils


def extract_sst_time(fp: Path) -> tuple[datetime, datetime]:
    """Extract the start and end dates from an SST file name.
    
    Args:
        fp (Path, str): Path to the SST file

    Returns:
        Tuple of start and end dates
    """
    matches = re.findall(r"(\d{4}-\d{2}-\d{2})", Path(fp).name)
    if len(matches) < 2:
        raise ValueError(f"Could not extract both start and end dates from filename: {fp.name}")
    start_date, end_date = matches[0], matches[1]
    return datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")


def filter_files_by_time(fps: list[Path], start_date: datetime | str | int, end_date: datetime | str | int) -> list[Path]:
    """Filter a list of SST file paths to only include those that fall within a given time range. Helpful for pre-loading files, even with dask.
    
    Args:
        fps (list[Path]): List of file paths to filter
        start_date (datetime, str, int): Start date of the time range
        end_date (datetime, str, int): End date of the time range
    """
    for fp in fps:
        start, end = extract_sst_time(fp)
        if start >= utils.parse_date(start_date) and end <= utils.parse_date(end_date):
            yield fp