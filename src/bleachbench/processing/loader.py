"""
Data loading utilities for patch-based NetCDF files organized by year.

This module provides convenient functions for lazy loading of downloaded patch data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import pandas as pd
import numpy as np
import xarray as xa
from dateutil.relativedelta import relativedelta
from rich.console import Console
import warnings

from bleachbench.utils import config


try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - optional dependency for non-ML users
    torch = None
    Dataset = object
        


def load_patches_by_year(
    base_dir: Path,
    var: str,
    years: Union[str, List[str]],
    patch_bounds: Optional[List[Tuple[float, float, float, float]]] = None,
    combine_patches: bool = True,
) -> Union[xa.Dataset, Dict[str, xa.Dataset]]:
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
                patch_ds = xa.open_mfdataset(nc_files, combine="by_coords")
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
                combined_ds = xa.combine_by_coords(year_datasets)
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
) -> xa.Dataset:
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



class SSTTimeSeriesLoader(Dataset):
    """
    Build per-observation SST time series for ML and analysis.

    For each row in `df` with columns: 'latitude', 'longitude', 'date',
    extracts a daily SST series for the window [date - window_months, date],
    by locating the appropriate patch directory and loading the necessary
    NetCDF files for the covered years.

    Returns PyTorch-ready tensors plus useful metadata for analysis.
    
    Args:
        df: DataFrame with columns: 'latitude', 'longitude', 'date'
        base_dir: Base directory containing patch subdirectories
        var: Variable name (e.g., "CRW_SST")
        window_months: Number of months to include before the observation date
        value_transform: Optional function to transform the SST values
        to_tensor: Whether to convert the SST values to PyTorch tensors
        interpolate_method: Interpolation method to use

    Returns:
        Dataset with columns: 'sst', 'time', 'lat', 'lon', 'date'
        and optional columns: 'bleach_presence', 'mean_percent_bleached'
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: Path = config.sst_dir / "CRW_SST",
        var: str = "CRW_SST",
        window_days_before: int = 31,
        # window_days_after: int = 31,
        value_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        to_tensor: bool = True,
        interpolate_method: str = "nearest",
    ) -> None:
        if torch is None and to_tensor:
            raise ImportError("PyTorch is required for to_tensor=True. Install torch or set to_tensor=False.")

        required_cols = {"latitude", "longitude", "date"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = df.reset_index(drop=True).copy()
        # ensure datetime
        if not np.issubdtype(self.df["date"].dtype, np.datetime64):
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        if self.df["date"].isna().any():
            raise ValueError("Some 'date' values could not be parsed to datetime.")

        self.base_dir = Path(base_dir)
        self.var = var
        self.window_days_before = int(window_days_before)
        self.value_transform = value_transform
        self.to_tensor = to_tensor
        self.interpolate_method = interpolate_method

        # cache patch directories listing for quick lookup
        self._patch_dirs = self._scan_patch_dirs(self.base_dir)
        if not self._patch_dirs:
            warnings.warn(f"No patch_*/ subdirectories found under {self.base_dir}. Dataset will be empty.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        obs_date = pd.Timestamp(row["date"])  # normalized

        start_date = (obs_date - relativedelta(days=self.window_days_before)).normalize()
        end_date = obs_date.normalize()

        # Locate patch that contains this point
        patch_dir, patch_bounds = self._find_patch_for_point(lat, lon)
        if patch_dir is None:
            raise FileNotFoundError(
                f"No patch directory containing point ({lat}, {lon}) found under {self.base_dir}"
            )
        
        # Validate that the required time window is available
        is_available, missing_years = self.validate_time_window(lat, lon, start_date, end_date)
        if not is_available:
            available_years = self.get_available_years_for_patch(lat, lon)
            # raise FileNotFoundError(  # TODO: should this exit?
            print(
                f"Time window {start_date.date()} to {end_date.date()} not fully available for point ({lat}, {lon}). "
                f"Missing years: {missing_years}. Available years: {available_years}"
            )
            return None

        # Load required years covering the date window
        years = sorted({str(d.year) for d in pd.date_range(start_date, end_date, freq="D")})
        # Patch directories store files directly inside, not nested by year
        all_files = sorted(patch_dir.glob(f"{self.var}_*.nc"))
        year_to_files: Dict[str, list] = {}
        for fp in all_files:
            name = fp.name
            # Expect patterns like CRW_SST_YYYY-...nc
            y = None
            try:
                # fast parse: first 4-digit number in name
                import re as _re
                m = _re.search(r"(19|20)\d{2}", name)
                if m:
                    y = m.group(0)
            except Exception:
                pass
            if y is not None:
                year_to_files.setdefault(y, []).append(fp)

        datasets = []
        missing_years = []
        for y in years:
            files = sorted(year_to_files.get(y, []))
            if not files:
                missing_years.append(y)
                continue
            try:
                ds = xa.open_mfdataset(files, combine="by_coords", combine_attrs="drop_conflicts")
                datasets.append(ds)
            except Exception as e:
                raise RuntimeError(f"Failed to open data for year {y} in {patch_dir}: {e}")

        if not datasets:
            raise FileNotFoundError(
                f"No NetCDF files for {self.var} found covering {start_date.date()}..{end_date.date()} in {patch_dir}. "
                f"Missing years: {missing_years}"
            )

        # Combine all years into a single dataset
        if len(datasets) == 1:
            full_ds = datasets[0]
        else:
            full_ds = xa.combine_by_coords(datasets)
            full_ds = full_ds.sortby('time')

        # Restrict to the exact time window
        window_ds = full_ds.sel(time=slice(start_date, end_date))

        # Ensure unique coords for interpolation
        window_ds = self._ensure_unique_coords(window_ds, ["latitude", "longitude"])  # type: ignore

        # Interpolate to point
        sst_da = window_ds[self.var]
        # Some datasets may use lowercase coords
        lat_name = "latitude" if "latitude" in sst_da.coords else ("lat" if "lat" in sst_da.coords else None)
        lon_name = "longitude" if "longitude" in sst_da.coords else ("lon" if "lon" in sst_da.coords else None)
        if lat_name is None or lon_name is None:
            raise KeyError("Dataset missing latitude/longitude coordinates.")

        sampled = sst_da.interp(
            {lat_name: xa.DataArray([lat], dims=("points",)),
             lon_name: xa.DataArray([lon], dims=("points",))},
            method=self.interpolate_method,
            kwargs={"fill_value": None},
        )

        series = sampled.isel(points=0).values  # (time,)
        time_index = pd.DatetimeIndex(window_ds.time.values)

        if self.value_transform is not None:
            series = self.value_transform(series)

        if self.to_tensor:
            x = torch.tensor(series, dtype=torch.float32)
        else:
            x = series  # numpy

        item: Dict[str, Any] = {
            "sst": x,  # shape (T,)
            "time": time_index,
            "lat": lat,
            "lon": lon,
            "date": obs_date,
        }

        # Optional labels if present
        if "bleach_presence" in row:
            item["bleach_presence"] = int(row["bleach_presence"]) if not pd.isna(row["bleach_presence"]) else None
        if "mean_percent_bleached" in row:
            val = row["mean_percent_bleached"]
            item["mean_percent_bleached"] = float(val) if not pd.isna(val) else None

        return item

    # ---------- Helpers ----------
    def get_available_years_for_patch(self, lat: float, lon: float) -> List[str]:
        """Get list of available years for a specific point (files are in patch dir)."""
        patch_dir, _ = self._find_patch_for_point(lat, lon)
        if patch_dir is None:
            return []
        available_years = set()
        import re as _re
        for fp in patch_dir.glob(f"{self.var}_*.nc"):
            # This regex searches for any 4-digit number that starts with '19' or '20' (i.e., any year from 1900 to 2099) in the file name.
            m = _re.search(r"(19|20)\d{2}", fp.name)
            if m:
                available_years.add(m.group(0))
        return sorted(available_years)
    
    def validate_time_window(self, lat: float, lon: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[bool, List[str]]:
        """
        Check if the required time window is available for a given point.
        
        Returns:
            (is_available, missing_years)
        """
        available_years = self.get_available_years_for_patch(lat, lon)
        required_years = {str(d.year) for d in pd.date_range(start_date, end_date, freq="D")}
        missing_years = sorted(required_years - set(available_years))
        return len(missing_years) == 0, missing_years
    
    def _scan_patch_dirs(self, base_dir: Path) -> List[Tuple[Path, Tuple[float, float, float, float]]]:
        patch_dirs = []
        if not base_dir.exists():
            return patch_dirs
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith("patch_"):
                parts = d.name.split("_")
                if len(parts) == 5:
                    try:
                        lat_min, lat_max, lon_min, lon_max = map(float, parts[1:])
                        patch_dirs.append((d, (lat_min, lat_max, lon_min, lon_max)))
                    except ValueError:
                        warnings.warn(f"Invalid patch directory name: {d.name}")
                        continue
        return patch_dirs

    def _find_patch_for_point(self, lat: float, lon: float) -> Tuple[Optional[Path], Optional[Tuple[float, float, float, float]]]:
        for d, (lat_min, lat_max, lon_min, lon_max) in self._patch_dirs:
            if (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max):
                return d, (lat_min, lat_max, lon_min, lon_max)
        return None, None

    def _ensure_unique_coords(self, ds: xa.Dataset, dims: List[str]) -> xa.Dataset:
        cleaned = ds
        for dim in dims:
            if dim in cleaned.dims:
                vals = cleaned[dim].values
                unique_vals, unique_idx = np.unique(vals, return_index=True)
                if len(unique_vals) != len(vals):
                    cleaned = cleaned.isel({dim: np.sort(unique_idx)})
        return cleaned
    
    def get_valid_indices(self, max_samples: int = None) -> List[int]:
        """
        Get list of valid indices without loading SST data.
        Only checks dataframe validity, not SST data availability.
        
        Args:
            max_samples: Maximum number of samples to return (for debugging)
            
        Returns:
            List of valid dataframe indices
        """
        # Check for required columns and valid data
        valid_mask = (
            self.df["latitude"].notna() & 
            self.df["longitude"].notna() & 
            self.df["date"].notna() & 
            self.df["bleach_presence"].notna()
        )
        
        valid_indices = self.df[valid_mask].index.tolist()
        
        # Sample if max_samples is specified
        if max_samples and len(valid_indices) > max_samples:
            rng = np.random.default_rng(42)
            sampled_indices = rng.choice(valid_indices, size=max_samples, replace=False)
            return sampled_indices.tolist()
        
        return valid_indices

    # ---------- Visualization Helper ----------
    @staticmethod
    def quick_plot(item: Dict[str, Any]):  # pragma: no cover - convenience
        import matplotlib.pyplot as plt
        y = item["sst"].cpu().numpy() if torch is not None and isinstance(item["sst"], torch.Tensor) else item["sst"]
        t = item["time"]
        plt.figure(figsize=(10, 3))
        plt.plot(t, y, label="SST")
        plt.axvline(item["date"], color="r", linestyle="--", label="observation")
        plt.title(f"SST @ ({item['lat']:.2f},{item['lon']:.2f})")
        plt.legend()
        plt.tight_layout()
        plt.show()

