"""
Data processing utilities for cleaning messy CSV columns.

This module provides a flexible framework for processing different types of messy data
commonly found in scientific datasets, particularly coral bleaching databases.
"""

import re
from datetime import datetime
from typing import Any, Callable, List

import numpy as np
import pandas as pd
import xarray as xa
import geopandas as gpd
from dask.diagnostics import ProgressBar

from bleachbench.utils import config

class DataProcessor:
    """
    A flexible data processor for cleaning messy CSV columns.

    This class provides a configurable approach to handle different types of messy data
    including ranges, units, special characters, and various data formats.
    """

    def __init__(self):
        self.processors = {
            "float_with_ranges": self._process_float_with_ranges,
            "depth": self._process_depth,
            "percentage": self._process_percentage,
        }

    def process_column(
        self, series: pd.Series, processor_type: str, **kwargs
    ) -> pd.Series:
        """
        Process a pandas Series using the specified processor type.

        Args:
            series: The pandas Series to process
            processor_type: The type of processor to use
            **kwargs: Additional arguments passed to the processor

        Returns:
            Processed pandas Series
        """
        if processor_type not in self.processors:
            raise ValueError(
                f"Unknown processor type: {processor_type}. "
                f"Available types: {list(self.processors.keys())}"
            )

        processor_func = self.processors[processor_type]
        return series.apply(lambda x: processor_func(x, **kwargs))

    def _process_float_with_ranges(self, x: Any, **kwargs) -> float:
        """
        Process values that may contain ranges and convert to float.

        Handles:
        - Regular numbers (int, float, string numbers)
        - Ranges like "10-20" (returns average)
        - NaN/None values

        Args:
            x: Value to process

        Returns:
            Float value or np.nan
        """
        if pd.isna(x):
            return np.nan, np.nan
        if isinstance(x, (int, float)):
            return float(x), float(x)
        if isinstance(x, str):
            x = x.strip()
            if "-" in x:
                parts = x.split("-")
                parts = [p.replace("%", "") for p in parts]
                try:
                    # Average of the two numbers
                    return (float(parts[0]), float(parts[1]))
                except Exception:
                    return np.nan, np.nan
            if "%" in x:
                x = x.split("%")[0]
            try:
                return float(x), float(x)
            except Exception:
                return np.nan, np.nan
        return np.nan, np.nan

    def _process_depth(self, x: Any, **kwargs) -> float:
        """
        Process depth values with complex formatting.

        Handles:
        - Regular numbers
        - Ranges (returns average)
        - Units (ft -> meters conversion)
        - Special characters (+, <, >, Ð)
        - Datetime objects (reconstructs range)
        - Various string formats

        Args:
            x: Value to process

        Returns:
            Depth in meters or np.nan
        """
        try:
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return float(x)

            # Handle datetime objects (Excel sometimes converts ranges to dates)
            if isinstance(x, datetime):
                return self._process_float_with_ranges(f"{x.month - 1}-{x.day}")

            if isinstance(x, str):
                x = x.lower().strip()

                # Remove common prefixes/suffixes
                x = x.replace("+", "").replace("<", "").replace(">", "")

                # Handle feet conversion
                if "ft" in x:
                    numbers = re.findall(r"[\d\.]+", x)
                    if len(numbers) == 1:
                        return float(numbers[0]) * 0.3048  # ft to meters
                    elif len(numbers) == 2:
                        # Average if range, then convert
                        return (float(numbers[0]) + float(numbers[1])) / 2 * 0.3048
                    else:
                        return np.nan

                # Clean up common issues
                x = x.replace("m", "")  # Remove meter indicator
                x = x.replace("Ð", "-")  # Fix encoding issue
                x = x.strip()

                # Process as float with ranges
                return self._process_float_with_ranges(x)

        except Exception as e:
            print(f"Error processing depth {x}: {e}")
            return np.nan

        return np.nan

    def _process_percentage(self, x: Any, **kwargs) -> tuple[float, float]:
        """
        Process percentage values.

        Similar to float_with_ranges but ensures values are within 0-100 range.

        Args:
            x: Value to process

        Returns:
            Tuple of minimum and maximum percentage values (0-100) or np.nan, np.nan
        """
        min_val, max_val = self._process_float_with_ranges(x)
        if not pd.isna(min_val) and not pd.isna(max_val):
            # Clamp to 0-100 range
            min_val = max(0, min(100, min_val))
            max_val = max(0, min(100, max_val))
        return min_val, max_val

    def add_processor(self, name: str, func: Callable) -> None:
        """
        Add a custom processor function.

        Args:
            name: Name of the processor
            func: Function that takes a value and returns processed value
        """
        self.processors[name] = func

    def process_df_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the metadata of a dataframe.
        """
        df.columns = df.columns.str.lower()
        df["date"] = pd.to_datetime(df["year"], format="%Y")
        return df

    def get_available_processors(self) -> List[str]:
        """Get list of available processor types."""
        return list(self.processors.keys())


# Example usage patterns
def process_bleaching_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example function showing how to process a bleaching dataframe.

    Args:
        df: Raw bleaching dataframe

    Returns:
        Processed dataframe
    # TODO: this can probably be folded in with the process_bleaching_dataframe_pipeline function
    """
    processor = DataProcessor()
    df_processed = df.copy()
    df_processed = processor.process_df_metadata(df_processed)

    # Process different columns with appropriate processors
    if "percent_bleached" in df_processed.columns:
        df_processed[["min_percent_bleached", "max_percent_bleached"]] = (
            processor.process_column(
                df_processed["percent_bleached"], "percentage"
            ).apply(pd.Series)
        )
        df_processed["mean_percent_bleached"] = (
            df_processed["min_percent_bleached"] + df_processed["max_percent_bleached"]
        ) / 2

    if "depth" in df_processed.columns:
        df_processed["depth"] = processor.process_column(df_processed["depth"], "depth")

    if "percent_mortality" in df_processed.columns:
        df_processed[["min_percent_mortality", "max_percent_mortality"]] = (
            processor.process_column(
                df_processed["percent_mortality"], "percentage"
            ).apply(pd.Series)
        )
        df_processed["mean_percent_mortality"] = (
            df_processed["min_percent_mortality"]
            + df_processed["max_percent_mortality"]
        ) / 2

    return df_processed


def sample_from_dataarray(
    da: xa.DataArray,
    coords: np.ndarray,
    chunks: dict = None,
    interp_method: str = "linear",
    fill_strategy: str = "mean",
    persist: bool = True,
) -> np.ndarray:
    """
    Interpolate a DataArray onto a set of (lon, lat) points.
    Returns a NumPy array of shape (N_points, N_time, ...) depending on input.

    Parameters
    ----------
    da (xa.DataArray):  The variable to interpolate, e.g., shape (time, lat, lon).
    coords (np.ndarray, shape (N_points, 2)):   Points to sample at, given as (lon, lat).
    chunks (dict, optional):    Optional rechunking of da before interpolation.
    interp_method (str):    Interpolation method passed to `xarray.DataArray.interp()`.
    fill_strategy (str or float):   Strategy for filling NaNs: 'mean', 'none', 'zero', or a numeric constant.
    persist (bool): Whether to persist the interpolated array in memory.

    Returns
    -------
    np.ndarray
        Interpolated values at (points, time, ...) depending on da.
    """
    if chunks:  # chunk if necessary
        da = da.chunk(chunks)
    
    # Check for and fix duplicate coordinates in the DataArray
    for dim in ['latitude', 'longitude']:
        if dim in da.dims:
            coord_values = da[dim].values
            if len(coord_values) != len(np.unique(coord_values)):
                # Remove duplicates by taking the first occurrence
                _, unique_idx = np.unique(coord_values, return_index=True)
                da = da.isel({dim: unique_idx})

    pts = xa.Dataset(
        {
            "latitude": (("points",), coords[:, 0]),
            "longitude": (("points",), coords[:, 1]),
        }
    )  # build tiny dataset with point coords

    # vectorized interpolation: result dims = (time, points)
    print("Sampling data...")
    sampled = da.interp(
        latitude=pts.latitude,
        longitude=pts.longitude,
        method=interp_method,
        kwargs={"fill_value": None},
    )

    if fill_strategy == "mean":
        sampled = sampled.fillna(da.mean(dim=["latitude", "longitude"]))
    elif fill_strategy == "zero":
        sampled = sampled.fillna(0.0)
    elif isinstance(fill_strategy, (float, int)):
        sampled = sampled.fillna(fill_strategy)
    elif fill_strategy == "none":
        pass  # leave NaNs
    else:
        raise ValueError(f"Unknown fill_strategy: {fill_strategy}")

    # 4) (Optional) persist in memory / show progress
    #    Wrap either persist() or compute() in a single ProgressBar
    with ProgressBar():
        sampled = sampled.persist() if persist else sampled.compute()

    # 5) Return a NumPy array in (points, time) order
    dims = list(sampled.dims)
    if "points" in dims:
        sampled = sampled.transpose("points", *[d for d in dims if d != "points"])

    return sampled.values


def load_combined_dataframe() -> pd.DataFrame:
    """Load the combined dataframe from CSV."""
    return pd.read_csv(config.bleaching_dir / "processed" / "combined_df.csv", low_memory=False)


def filter_essential_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Essential filtering: remove NaNs in critical columns."""
    print("Filtering essential NaNs...")
    # filter nans in month and year
    df = df[df["month"].notna() & df["year"].notna()]
    # filter nans in latitude and longitude
    df = df[df["latitude"].notna() & df["longitude"].notna()]
    return df


def filter_bleaching_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Filter NaNs in bleaching data."""
    print("Filtering bleaching NaNs...")
    return df[df["mean_percent_bleached"].notna()]


def filter_reef_areas(df: pd.DataFrame) -> pd.DataFrame:
    """Filter points to only include reef areas using UNEP-GCDR data."""
    print("Loading UNEP-GCDR dataframe...")
    unep_gdcr_fp = config.unep_gdcr_dir / "WCMC008_CoralReef2021_Py_v4_1.shp"
    unep_gdcr_df = gpd.read_file(unep_gdcr_fp)

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    )
    unep_gdcr_df = unep_gdcr_df.to_crs(gdf_points.crs)
    print("Joining dataframes...")
    reef_points = gpd.sjoin(
        gdf_points,
        unep_gdcr_df[["geometry"]],
        how="inner",
    )
    return reef_points


def add_bleach_presence_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add bleaching presence labels based on thresholds."""
    print("Adding bleaching presence labels...")
    # 3.1 Mark absence of ecologically significant bleaching as maximum observation <=10%
    df["bleach_presence"] = df["max_percent_bleached"].apply(
        lambda x: 0 if x <= 10 else 1
    )
    # 3.2 Mark presence of ecologically significant bleaching for minimum observation >=20%
    df["bleach_presence"] = df["min_percent_bleached"].apply(
        lambda x: 1 if x >= 20 else 0
    )    
    return df

def remove_uncertain_bleach_presence(df: pd.DataFrame) -> pd.DataFrame:
    """Remove observations which have max estimation > 10% while minimum < 20%"""
    print("Removing uncertain bleach presence...")
    uncertain_indices = (
        df["max_percent_bleached"] > 10) & (df["min_percent_bleached"] < 20)
    return df.loc[~uncertain_indices]

def filter_years_2017(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out years after 2017."""
    print("Filtering years after 2017...")
    return df[df["year"] <= 2017]


def filter_years_by_observation_count(df: pd.DataFrame, obs_threshold: int = 100) -> pd.DataFrame:
    """Retain only years with sufficient observations and all preceding years."""
    print(f"Filtering years with <{obs_threshold} observations...")
    init_count = len(df)
    no_obs_per_year = df.groupby("year").size()

    valid_years = []
    years_sorted = sorted(no_obs_per_year.index)

    valid = True
    for year in reversed(years_sorted):
        if no_obs_per_year.loc[year] >= obs_threshold and valid:
            valid_years.append(year)
        else:
            valid = False
            break

    filtered_df = df[df["year"].isin(valid_years)]
    print(f"Removed {init_count - len(filtered_df)} observations due to insufficient yearly data")
    return filtered_df


def determine_mmm(df: pd.DataFrame) -> pd.DataFrame:
    """Determine climatology"""
    month_of_mmm = xa.open_dataset(config.crw_sst_dir / "processed" / "month_of_mmm.nc")
    coords = np.vstack((df["longitude"], df["latitude"])).T
    mmm_i = sample_from_dataarray(month_of_mmm.prediction, coords, interp_method="nearest", fill_strategy="none")
    df["mmm_i"] = mmm_i
    return df

def filter_mmm_exposure_period(df: pd.DataFrame) -> pd.DataFrame:
    """Filter points to only include points within the MMM exposure period."""
    print("Filtering MMM exposure period...")
    df = determine_mmm(df)
    df = filter_missing_sst_data(df)
    
    # remove points where mmm_i is less than mmm_min_exposure_period
    print("Removed", len(df) - len(df[df["month"] >= (df["mmm_i"] - 1)]), "points where mmm_i is less than mmm_min_exposure_period")
    df = df[df["month"] >= (df["mmm_i"] - 1)]
    # remove points where mmm_i is greater than mmm_max_exposure_period
    print("Removed", len(df) - len(df[df["month"] <= (df["mmm_i"] + 3)]), "points where mmm_i is greater than mmm_max_exposure_period")
    df = df[df["month"] <= (df["mmm_i"] + 3)]
    return df


def filter_missing_sst_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter points to only include points with SST data."""
    print("Filtering missing SST data...")
    no_nan_df = df[df["mmm_i"].notna()]
    print(f"Removed {len(df) - len(df[df['mmm_i'].notna()])} points where mmm_i is nan")
    return no_nan_df


def process_bleaching_dataframe_pipeline(
    steps: list[str] = ["remove_uncertain_bleach_presence"],
    obs_threshold: int = 100,
    do_essential_steps: bool = True,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Process bleaching dataframe with configurable pipeline steps. Defaults to applying all essential steps.
    
    Args:
        steps (list[str]): List of processing steps to apply. If None, applies all steps.
        obs_threshold (int): Minimum observations per year for year filtering
        do_essential_steps (bool): Whether to apply essential steps
        verbose (bool): Whether to print verbose output
        **kwargs (dict): Additional arguments passed to individual steps
        

    Returns:
        pd.DataFrame: Processed dataframe
    """
    available_steps = {
        'essential_nans': filter_essential_nans,
        'bleaching_nans': filter_bleaching_nans,
        'reef_areas': filter_reef_areas,
        'bleach_labels': add_bleach_presence_labels,
        'remove_uncertain_bleach_presence': remove_uncertain_bleach_presence,
        'years_2017': filter_years_2017,
        'years_obs_count': lambda df: filter_years_by_observation_count(df, obs_threshold),
        'determine_mmm': determine_mmm,
        'filter_missing_sst_data': filter_missing_sst_data,
        'filter_mmm_exposure_period': filter_mmm_exposure_period,
    }
    
    # Load initial data
    df = load_combined_dataframe()
    print(f"Starting with {len(df)} observations\n")
    
    # Define available steps
    essential_steps = ["essential_nans", "bleaching_nans", "determine_mmm", "filter_missing_sst_data", "bleach_labels", "years_2017", "years_obs_count"]
    
    # do essential steps
    if do_essential_steps:
        for step_name in essential_steps:
            df = available_steps[step_name](df)
            print(f"After {step_name}: {len(df)} observations\n") if verbose else None
            # print(f"After {step_name}: {len(df)} observations")
    
    # do additional steps
    if steps is not None:
        for step_name in steps:
            if step_name not in available_steps:
                raise ValueError(f"Unknown step: {step_name}. Available: {list(available_steps.keys())}")
            if do_essential_steps and step_name in essential_steps:
                continue
            
            df = available_steps[step_name](df)
            print(f"After {step_name}: {len(df)} observations\n") if verbose else None

    
    print(f"\nFinal dataset: {len(df)} observations")
    return df