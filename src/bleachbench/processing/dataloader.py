# general
# Advanced implementation with UI progress tracking
# add argparse to allow command line input
import argparse
import concurrent.futures
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from dateutil.relativedelta import relativedelta
from rich import print as rich_print

from bleachbench.utils import config, ui, utils


class GetERDAPP:
    def __init__(
        self,
        var: str,
        date_range: tuple[int, int],
        lats: tuple[float, float],
        lons: tuple[float, float],
        output_dir: Path,
        max_workers: int,
        retry_failed: bool,
        max_retries: int,
        patch_width: float = 20,
    ):
        self.var = var
        self.date_range = date_range
        self.lats = lats
        self.lons = lons
        self.output_dir = self.make_output_dir(output_dir)

        self.max_workers = max_workers
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        self.patch_width = patch_width
        # TODO:
        # estimate file size/download time
        # summary of download after finished (number saved, failed, time taken)

    def calc_area(self) -> float:
        lat_side = abs(max(self.lats) - min(self.lats))
        lon_side = abs(max(self.lons) - min(self.lons))
        return lat_side * lon_side

    def require_patchwork(self) -> bool:
        """Always use patchwork downloading for consistency."""
        return True

    def make_output_dir(self, output_dir: Path) -> Path:
        """Name the output directory based on the spatial extent"""
        # Create a directory name based on the full spatial extent
        output_dir_fp = (
            output_dir
            / f"{min(self.lats):.0f}_{max(self.lats):.0f}_{min(self.lons):.0f}_{max(self.lons):.0f}"
        )
        output_dir_fp.mkdir(parents=True, exist_ok=True)
        return output_dir_fp

    def make_patch_output_dir(
        self,
        output_dir: Path,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Path:
        """Create output directory for a specific patch"""
        patch_dir = (
            output_dir
            # / f"{lat_min:.0f}_{lat_max:.0f}_{lon_min:.0f}_{lon_max:.0f}"
            / f"patch_{lat_min:.0f}_{lat_max:.0f}_{lon_min:.0f}_{lon_max:.0f}"
        )
        patch_dir.mkdir(parents=True, exist_ok=True)
        return patch_dir

    def get_patch_bounds(
        self,
    ) -> list[tuple[float, float, float, float]]:
        """Generate patch bounds for patchwork downloading.

        Snaps patch bounds to global grid intervals starting from zero.
        For example, if patch_width=20 and requested bounds are (5, 35), (10, 45),
        this will generate patches: (0, 20), (20, 40) for lat and (0, 20), (20, 40) for lon,
        then intersect with the requested bounds.
        """
        patches = []

        # Snap to global grid starting from zero
        # Find the grid-aligned bounds that encompass our requested area
        lat_grid_min = np.floor(min(self.lats) / self.patch_width) * self.patch_width
        lat_grid_max = np.ceil(max(self.lats) / self.patch_width) * self.patch_width
        lon_grid_min = np.floor(min(self.lons) / self.patch_width) * self.patch_width
        lon_grid_max = np.ceil(max(self.lons) / self.patch_width) * self.patch_width

        # Generate patches on the global grid
        lat_current = lat_grid_min
        while lat_current < lat_grid_max:
            lon_current = lon_grid_min
            while lon_current < lon_grid_max:
                # Define patch bounds on the global grid
                patch_lat_min = lat_current
                patch_lat_max = lat_current + self.patch_width
                patch_lon_min = lon_current
                patch_lon_max = lon_current + self.patch_width

                # Check if this patch intersects with our requested bounds
                intersect_lat_min = max(patch_lat_min, min(self.lats))
                intersect_lat_max = min(patch_lat_max, max(self.lats))
                intersect_lon_min = max(patch_lon_min, min(self.lons))
                intersect_lon_max = min(patch_lon_max, max(self.lons))

                # Only add patch if there's actual intersection
                if (
                    intersect_lat_max > intersect_lat_min
                    and intersect_lon_max > intersect_lon_min
                ):
                    # Use the full grid-aligned patch bounds, not the intersection
                    # This ensures consistent global tiling
                    patches.append(
                        (patch_lat_min, patch_lat_max, patch_lon_min, patch_lon_max)
                    )

                lon_current += self.patch_width
            lat_current += self.patch_width

        return patches

    def print_patch_info(self):
        """Print information about the patches that will be created."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        patches = self.get_patch_bounds()

        console.print(
            f"\n[bold]Patch Information for {self.calc_area():.1f} sq degree area:[/bold]"
        )
        console.print(
            f"Original bounds: {min(self.lats):.1f}°-{max(self.lats):.1f}°N, {min(self.lons):.1f}°-{max(self.lons):.1f}°E"
        )
        console.print(
            f"Number of {self.patch_width:.0f}x{self.patch_width:.0f}° patches: {len(patches)}"
        )

        table = Table(title="Patch Bounds")
        table.add_column("Patch #", style="cyan")
        table.add_column("Lat Min", style="green")
        table.add_column("Lat Max", style="green")
        table.add_column("Lon Min", style="blue")
        table.add_column("Lon Max", style="blue")

        for i, (lat_min, lat_max, lon_min, lon_max) in enumerate(patches, 1):
            table.add_row(
                str(i),
                f"{lat_min:.1f}°",
                f"{lat_max:.1f}°",
                f"{lon_min:.1f}°",
                f"{lon_max:.1f}°",
            )

        console.print(table)

    def get_erdapp_thredds_nc_file_urls(
        self,
    ) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Generate ERDDAP THREDDS NetCDF URLs for patch chunks.

        Returns:
            List of tuples (url, patch_bounds) where patch_bounds is (lat_min, lat_max, lon_min, lon_max)
        """
        start_date = utils.parse_date(self.date_range[0])
        end_date = utils.parse_date(self.date_range[1])

        if start_date > end_date:
            raise ValueError("Start date must be before end date.")

        # ERDDAP query components
        url_base = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.nc?"

        patches = self.get_patch_bounds()
        urls_with_bounds = []

        for patch_bounds in patches:
            lat_min, lat_max, lon_min, lon_max = patch_bounds
            spatial_part = f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"

            current_start = start_date.replace(hour=12, minute=0, second=0)
            while current_start < end_date:
                # End of current month or end_date, whichever comes first
                next_month = (current_start + relativedelta(months=1)).replace(day=1)
                current_end = min(next_month - timedelta(days=1), end_date)

                # Build the time slice
                time_slice = f"[({current_start.strftime('%Y-%m-%dT%H:%M:%SZ')}):1:({current_end.strftime('%Y-%m-%dT%H:%M:%SZ')})]"
                query = f"{self.var}{quote(time_slice + spatial_part)}"
                url = f"{url_base}{query}"

                urls_with_bounds.append((url, patch_bounds))
                current_start = next_month

        return urls_with_bounds

    def remove_part_files(self, main_dir: Path) -> None:
        """Remove part files from the main directory. Cleans up temporary/unfinished files after downloads."""
        part_files = list(main_dir.glob("*.part"))
        if not part_files:
            rich_print("No .part files found to remove.")
            return

        for part_file in part_files:
            try:
                part_file.unlink()
                rich_print(f"Removed .part file: {part_file}")
            except Exception as e:
                rich_print(f"Failed to remove {part_file}: {e}")

    def get_output_fp(
        self, url: str, patch_bounds: tuple[float, float, float, float] = None
    ) -> Path:
        """Generate the output file path for a given URL."""
        import re

        matches = re.findall(r"(\d{4}-\d{2}-\d{2})T", url)
        if matches:
            date_str = "_".join(matches)
        else:
            date_str = "unknown"

        if patch_bounds is not None:
            # For patchwork downloads, create patch-specific subdirectory
            lat_min, lat_max, lon_min, lon_max = patch_bounds
            patch_dir = self.make_patch_output_dir(
                self.output_dir.parent, lat_min, lat_max, lon_min, lon_max
            )
            return patch_dir / f"{self.var}_{date_str}.nc"
        else:
            # For single downloads, use the main output directory
            return self.output_dir / f"{self.var}_{date_str}.nc"

    def file_exists(
        self, url: str, patch_bounds: tuple[float, float, float, float] = None
    ) -> bool:
        """Check if a file already exists based on its URL."""
        file_path = self.get_output_fp(url, patch_bounds)
        exists = file_path.exists() and file_path.stat().st_size > 0
        if not exists:
            # clean up any leftover .part files
            part_file = file_path.with_suffix(file_path.suffix + ".part")
            if part_file.exists():
                try:
                    part_file.unlink()
                except Exception:
                    pass
        return exists

    def prune_urls(
        self, urls_with_bounds: list[tuple[str, tuple[float, float, float, float]]]
    ) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Prune URLs with patch bounds that already exist in the output directory."""
        return [
            (url, bounds)
            for url, bounds in urls_with_bounds
            if not self.file_exists(url, bounds)
        ]

    def _process_url_with_ui(self, url_data, ui_instance):
        """Process a single URL with the UI."""
        try:
            # Handle both single URLs and (url, patch_bounds) tuples
            if isinstance(url_data, tuple):
                url, patch_bounds = url_data
            else:
                url = url_data
                patch_bounds = None

            file_id = url  # use url as unique identifier
            ui_instance.set_status(file_id, "STARTING", "cyan")
            success = self._download_url_direct_ui(url, ui_instance, patch_bounds)
            if success:
                ui_instance.set_status(file_id, "DONE", "green")
                ui_instance.complete_file(file_id)
                # Hide the file's progress bar after 5 seconds
            else:
                ui_instance.set_status(file_id, "FAILED", "red")
                ui_instance.complete_file(file_id)
            self.hide_file_after_delay(file_id, ui_instance, 5)
        except Exception as e:
            ui_instance.set_status(file_id, "ERROR", "red")
            ui_instance.add_failed(file_id, f"Error: {e}", e)
            ui_instance.complete_file(file_id)

    def _download_url_direct_ui(self, url, ui_instance, patch_bounds=None):
        """Directly download a file from a URL using the UI."""
        file_path = self.get_output_fp(url, patch_bounds)
        temp_path = file_path.with_suffix(file_path.suffix + ".part")
        file_id = url
        try:
            ui_instance.set_status(file_id, "DOWNLOADING", "blue")
            if temp_path.exists():
                time.sleep(2)
                if file_path.exists() and file_path.stat().st_size > 0:
                    return True
            response = requests.get(url, stream=True, timeout=(30, 300))
            response.raise_for_status()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if temp_path.exists():
                temp_path.unlink()
            total = int(response.headers.get("content-length", 0))
            bytes_downloaded = 0
            # set the per-file progress bar total to the file size (if available)
            ui_instance.update_file_progress(file_id, 0, total)
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        ui_instance.update_file_progress(
                            file_id, bytes_downloaded, total
                        )
            if temp_path.exists() and temp_path.stat().st_size > 0:
                temp_path.rename(file_path)
                return True
        except Exception as e:
            print(f"Direct download failed for {file_path.name}: {e}")
            for path in [temp_path, file_path]:
                if path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass
        return False

    def hide_file_after_delay(self, file_id, ui_instance, delay_seconds):
        """Hide the progress bar for a file after a delay."""
        import threading

        def hide_file():
            time.sleep(delay_seconds)
            try:
                if hasattr(ui_instance, "hide_file"):
                    ui_instance.hide_file(file_id)
                elif hasattr(ui_instance, "file_task_ids") and hasattr(
                    ui_instance, "progress"
                ):
                    task_id = ui_instance.file_task_ids.get(file_id)
                    if task_id is not None:
                        ui_instance.progress.update(task_id, visible=False)
            except Exception:
                pass

        t = threading.Thread(target=hide_file, daemon=True)
        t.start()

    def run(self):
        """Run the downloader using patchwork approach."""
        from rich.console import Console

        console = Console()
        self.remove_part_files(self.output_dir)

        console.print(
            f"[blue]Using patchwork download scheme for {self.calc_area():.1f} sq degree area.[/blue]"
        )
        console.print(
            f"Generating patch grid for {self.patch_width:.0f}x{self.patch_width:.0f} degree tiles..."
        )

        # Show patch information
        self.print_patch_info()

        urls_with_bounds = self.get_erdapp_thredds_nc_file_urls()
        console.print(f"Total URLs to process: {len(urls_with_bounds)}")

        urls_to_download = self.prune_urls(urls_with_bounds)

        if not urls_to_download:
            console.print("All patch files already exist, nothing to download.")
            return

        dl_str = "patch file" if len(urls_to_download) == 1 else "patch files"
        console.print(f"Downloading {len(urls_to_download)} new {dl_str}...")

        # Extract just URLs for the UI (it expects a list of strings)
        urls_for_ui = [url for url, _ in urls_to_download]

        with ui.DownloadProgressUI(urls_for_ui) as ui_instance:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                _ = [
                    executor.submit(self._process_url_with_ui, url_data, ui_instance)
                    for url_data in urls_to_download
                ]
            ui_instance.print_summary()

        end_time = time.localtime()
        console.print(f"\n:clock3: END: {time.strftime('%Y-%m-%d %H:%M:%S', end_time)}")

        # TODO: Implement retry logic for patchwork downloads
        # Note: This would need to be updated to handle the different data structures


def main():
    parser = argparse.ArgumentParser(
        description="Download ERDDAP THREDDS NetCDF files."
    )
    parser.add_argument(
        "--var",
        type=str,
        default="CRW_SST",
        help="Variable to download (e.g., CRW_SST, CRW_DHW).",
    )
    parser.add_argument(
        "--date-range",
        dest="date_range",
        nargs=2,
        type=str,
        default=["2012", "2014"],
        metavar=("START", "END"),
        help="Date range (two values). Accepts YYYY, YYYYMM, YYYYMMDD, or ISO8601.",
    )
    parser.add_argument(
        "--lats",
        nargs=2,
        type=float,
        default=[-32.0, 0.0],
        metavar=("MIN_LAT", "MAX_LAT"),
        help="Latitude bounds (min max).",
    )
    parser.add_argument(
        "--lons",
        nargs=2,
        type=float,
        default=[130.0, 170.0],
        metavar=("MIN_LON", "MAX_LON"),
        help="Longitude bounds (min max).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=(config.sst_dir / "CRW_SST"),
        help="Directory to write NetCDF files.",
    )
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=int,
        default=3,
        help="Maximum concurrent downloads.",
    )
    parser.add_argument(
        "--retry-failed",
        dest="retry_failed",
        action="store_true",
        default=False,
        help="Retry failed downloads after initial pass.",
    )
    parser.add_argument(
        "--max-retries",
        dest="max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed downloads.",
    )
    parser.add_argument(
        "--show-patches",
        dest="show_patches",
        action="store_true",
        default=False,
        help="Show patch information without downloading (useful for large areas).",
    )

    args = parser.parse_args()

    erdapp = GetERDAPP(
        var=args.var,
        date_range=(args.date_range[0], args.date_range[1]),
        lats=(args.lats[0], args.lats[1]),
        lons=(args.lons[0], args.lons[1]),
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        retry_failed=args.retry_failed,
        max_retries=args.max_retries,
    )

    if args.show_patches:
        erdapp.print_patch_info()
    else:
        erdapp.run()


if __name__ == "__main__":
    main()


# DEPRECATED
# def download_erdapp_thredds_nc_files(
#     var: str = "CRW_SST",
#     date_range: tuple[str, str] = ("1985-04-01T12:00:00Z", "2024-04-28T12:00:00Z"),
#     lats: tuple[float, float] = (-32, 0),
#     lons: tuple[float, float] = (130, 170),
#     output_dir=None,
# ):
#     """Download variable data from ERDDAP server for a given date range and spatial extent.

#     ERDDAP link: https://pae-paha.pacioos.hawaii.edu/thredds/satellite.html?dataset=dhw_5km

#     Parameters
#     var (str): The variable to download. Default is "CRW_SST".
#     date_range (tuple): A tuple of strings representing the start and end dates in the format "YYYY-MM-DDTHH:MM:SSZ".
#     lats (tuple): A tuple of floats representing the minimum and maximum latitudes to download.
#     lons (tuple): A tuple of floats representing the minimum and maximum longitudes to download.
#     output_dir (Path): The directory to save the downloaded files to.

#     var options:
#     N.B. "var" value is just captilised portion (e.g. enter "CRW_SST" for "CRW_SST (Celsius)")
#         CRW_SST (Celsius) = nighttime sea surface temperature = sea_surface_temperature
#         CRW_BAA (1) = coral bleaching alert area = coral_bleaching_alert_area
#         CRW_BAA_7D_MAX (1) = coral bleaching alert area 7-day maximum = coral_bleaching_alert_area_7day_maximum
#         CRW_DHW (Celsius weeks) = coral bleaching degree heating week = coral_bleaching_degree_heating_week
#         CRW_HOTSPOT (Celsius) = coral bleaching hot spots = coral_bleaching_hot_spots
#         CRW_SSTANOMALY (Celsius) = nighttime sea surface temperature anomaly = sea_surface_temperature_anomaly
#     """
#     # set correct base URL for the variable
#     url_base = f"https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.nc?{var}%5B"
#     # set spatial constraints
#     url_end = (
#         f"%5D%5B({min(lats)}):1:({max(lats)})%5D%5B({min(lons)}):1:({max(lons)})%5D"
#     )

#     if not output_dir:  # if no output directory path provided
#         output_dir = Path(config.sst_dir) / var
#         output_dir.mkdir(parents=True, exist_ok=True)

#     # if date_range is a tuple of strings, convert start and end dates to datetime objects
#     if isinstance(date_range[0], str):
#         dates = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ") for date in date_range]
#     elif isinstance(date_range[0], datetime):
#         dates = date_range

#     # assign start_date to oldest date
#     start_date = min(dates)
#     end_date = max(dates)

#     print(
#         f"Downloading {var} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} to {output_dir}\n"
#     )
#     # Iterate over month-by-month chunks
#     while start_date < end_date:
#         # Calculate the end of the month
#         next_month = start_date + relativedelta(months=1)
#         end_of_month = next_month.replace(day=1) - relativedelta(days=1)

#         # Construct the URL for the current month
#         month_url = f"{url_base}({start_date.isoformat()}Z):1:({end_of_month.isoformat()}Z){url_end}"

#         output_fp = output_dir / f"{var}_{start_date.strftime('%Y-%m')}.nc"
#         if not output_fp.exists():
#             # Download the data
#             response = requests.get(month_url)
#             if response.status_code == 200:  # successful request
#                 data = xa.open_dataset(response.content)
#                 data.to_netcdf(output_fp)
#                 print(
#                     f"Downloaded data for {start_date.strftime('%Y-%m-%d')} to {end_of_month.strftime('%Y-%m-%d')} "
#                     f"and saved to {output_fp}"
#                 )
#             elif response.status_code == 404:
#                 print(
#                     f"Data for {start_date.strftime('%Y-%m-%d')} to {end_of_month.strftime('%Y-%m-%d')} not found ({response.status_code}) at {response.url}.\nAre you sure it exists?"
#                 )
#             else:
#                 print(
#                     f"Failed to download data for {start_date.strftime('%Y-%m-%d')} to {end_of_month.strftime('%Y-%m-%d')}.\nError: \n{response.text}"
#                 )
#         else:
#             print(
#                 f"\tData for {start_date.strftime('%Y-%m-%d')} to {end_of_month.strftime('%Y-%m-%d')} already exists. Skipping download..."
#             )
#         # iterate to next month
#         start_date = next_month

#     print("\nDownload complete.")
