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
