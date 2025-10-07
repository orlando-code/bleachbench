"""
Data loading utilities for patch-based NetCDF files organized by year.

This module provides convenient functions for lazy loading of downloaded patch data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import xarray as xr
from rich.console import Console


def load_patches_by_year(
    base_dir: Path,
    var: str,
    years: Union[str, List[str]],
    patch_bounds: Optional[List[Tuple[float, float, float, float]]] = None,
    combine_patches: bool = True,
) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
    """
    Lazy load patch data for specific years.

    Args:
        base_dir: Base directory containing patch subdirectories
        var: Variable name (e.g., "CRW_SST")
        years: Year(s) to load. Can be single year string or list of years
        patch_bounds: Optional list of patch bounds to load. If None, loads all available patches
        combine_patches: If True, combine all patches into single dataset. If False, return dict by year

    Returns:
        xarray Dataset (if combine_patches=True) or dict of datasets by year
    """
    if isinstance(years, str):
        years = [years]

    console = Console()
    datasets_by_year = {}

    for year in years:
        console.print(f"Loading data for year {year}...")
        year_datasets = []

        # Find all patch directories if not specified
        if patch_bounds is None:
            patch_dirs = [
                d
                for d in base_dir.iterdir()
                if d.is_dir() and d.name.startswith("patch_")
            ]
            patch_bounds_to_load = []
            for patch_dir in patch_dirs:
                # Extract bounds from directory name: patch_lat1_lat2_lon1_lon2
                parts = patch_dir.name.split("_")
                if len(parts) == 5:  # patch_lat1_lat2_lon1_lon2
                    try:
                        lat_min, lat_max, lon_min, lon_max = map(float, parts[1:])
                        patch_bounds_to_load.append(
                            (lat_min, lat_max, lon_min, lon_max)
                        )
                    except ValueError:
                        continue
        else:
            patch_bounds_to_load = patch_bounds

        # Load data from each patch
        for lat_min, lat_max, lon_min, lon_max in patch_bounds_to_load:
            patch_dir = (
                base_dir
                / f"patch_{lat_min:.0f}_{lat_max:.0f}_{lon_min:.0f}_{lon_max:.0f}"
                / year
            )

            if not patch_dir.exists():
                console.print(
                    f"  [yellow]Warning: No data found for patch ({lat_min}, {lat_max}, {lon_min}, {lon_max}) in {year}[/yellow]"
                )
                continue

            # Load all NetCDF files for this patch and year
            nc_files = list(patch_dir.glob(f"{var}_*.nc"))
            if not nc_files:
                console.print(
                    f"  [yellow]Warning: No {var} files found in {patch_dir}[/yellow]"
                )
                continue

            try:
                # Load and combine monthly files for this patch
                patch_ds = xr.open_mfdataset(nc_files, combine="by_coords")
                # Add patch bounds as attributes for reference
                patch_ds.attrs.update(
                    {
                        "patch_lat_min": lat_min,
                        "patch_lat_max": lat_max,
                        "patch_lon_min": lon_min,
                        "patch_lon_max": lon_max,
                    }
                )
                year_datasets.append(patch_ds)
                console.print(
                    f"  ✓ Loaded patch ({lat_min:.0f}, {lat_max:.0f}, {lon_min:.0f}, {lon_max:.0f})"
                )
            except Exception as e:
                console.print(
                    f"  [red]Error loading patch ({lat_min}, {lat_max}, {lon_min}, {lon_max}): {e}[/red]"
                )

        if year_datasets:
            if combine_patches:
                # Combine all patches for this year
                combined_ds = xr.combine_by_coords(year_datasets)
                datasets_by_year[year] = combined_ds
                console.print(f"  ✓ Combined {len(year_datasets)} patches for {year}")
            else:
                datasets_by_year[year] = year_datasets
        else:
            console.print(f"  [red]No data loaded for year {year}[/red]")

    if len(years) == 1 and combine_patches:
        return datasets_by_year.get(years[0])
    else:
        return datasets_by_year


def get_available_data_summary(base_dir: Path, var: str) -> pd.DataFrame:
    """
    Get a summary of available data as a pandas DataFrame.

    Args:
        base_dir: Base directory containing patch subdirectories
        var: Variable name (e.g., "CRW_SST")

    Returns:
        DataFrame with columns: patch_bounds, lat_min, lat_max, lon_min, lon_max, available_years, file_count
    """
    summary_data = []

    # Find all patch directories
    patch_dirs = [
        d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("patch_")
    ]

    for patch_dir in patch_dirs:
        # Extract bounds from directory name
        parts = patch_dir.name.split("_")
        if len(parts) != 5:  # patch_lat1_lat2_lon1_lon2
            continue

        try:
            lat_min, lat_max, lon_min, lon_max = map(float, parts[1:])
        except ValueError:
            continue

        # Find available years
        year_dirs = [d for d in patch_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        available_years = []
        total_files = 0

        for year_dir in year_dirs:
            nc_files = list(year_dir.glob(f"{var}_*.nc"))
            if nc_files:
                available_years.append(year_dir.name)
                total_files += len(nc_files)

        summary_data.append(
            {
                "patch_bounds": f"({lat_min:.0f}, {lat_max:.0f}, {lon_min:.0f}, {lon_max:.0f})",
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "available_years": sorted(available_years),
                "file_count": total_files,
            }
        )

    return pd.DataFrame(summary_data)


def load_specific_months(
    base_dir: Path,
    var: str,
    year: str,
    months: Union[int, List[int]],
    patch_bounds: Optional[List[Tuple[float, float, float, float]]] = None,
) -> xr.Dataset:
    """
    Load specific months from a year.

    Args:
        base_dir: Base directory containing patch subdirectories
        var: Variable name (e.g., "CRW_SST")
        year: Year to load from
        months: Month(s) to load (1-12). Can be single int or list
        patch_bounds: Optional list of patch bounds to load

    Returns:
        xarray Dataset containing only the specified months
    """
    if isinstance(months, int):
        months = [months]

    # Load the full year first
    year_data = load_patches_by_year(
        base_dir, var, year, patch_bounds, combine_patches=True
    )

    if year_data is None:
        return None

    # Filter to specific months
    month_data = year_data.sel(time=year_data.time.dt.month.isin(months))

    return month_data


# Example usage functions
def example_lazy_loading():
    """Example of how to use the lazy loading functions."""
    from bleachbench.utils import config

    base_dir = config.sst_dir / "your_area_bounds"  # Replace with actual bounds

    # Load single year, all patches
    sst_2020 = load_patches_by_year(base_dir, "CRW_SST", "2020")

    # Load multiple years
    sst_multi = load_patches_by_year(base_dir, "CRW_SST", ["2020", "2021", "2022"])

    # Load specific patches only
    patch_bounds = [(0.0, 20.0, 0.0, 20.0), (20.0, 40.0, 0.0, 20.0)]
    sst_patches = load_patches_by_year(base_dir, "CRW_SST", "2020", patch_bounds)

    # Load specific months
    summer_2020 = load_specific_months(
        base_dir, "CRW_SST", "2020", [6, 7, 8]
    )  # Jun-Aug

    # Get data summary
    summary = get_available_data_summary(base_dir, "CRW_SST")
    print(summary)

    return sst_2020, sst_multi, sst_patches, summer_2020, summary
