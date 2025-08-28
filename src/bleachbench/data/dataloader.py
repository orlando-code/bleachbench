# general
# custom
# from src.config import config

# get  list of file urls to download
import concurrent.futures
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import requests
from dateutil.relativedelta import relativedelta
from rich import print as rich_print

from bleachbench.utils import config, ui


class GetERDAPP:
    def __init__(
        self,
        var: str = "CRW_SST",
        date_range: tuple[str | datetime, str | datetime] = (
            # "2014-01-01T12:00:00Z",
            # "2018-01-28T12:00:00Z",
            "2012-01-01T12:00:00Z",
            "2014-01-28T12:00:00Z",
        ),
        lats: tuple[float, float] = (-32, 0),
        lons: tuple[float, float] = (130, 170),
        output_dir: Path = config.sst_dir / "CRW_SST",
        max_workers: int = 3,
    ):
        self.var = var
        self.date_range = date_range
        self.lats = lats
        self.lons = lons
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.max_workers = max_workers

    def get_erdapp_thredds_nc_file_urls(self) -> list[str]:
        """Generate ERDDAP THREDDS NetCDF URLs for monthly chunks over a date range and lat-lon bounds."""

        # Parse dates if given as strings
        def parse_date(d):
            return (
                datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ") if isinstance(d, str) else d
            )

        start_date = parse_date(self.date_range[0])
        end_date = parse_date(self.date_range[1])

        if start_date > end_date:
            raise ValueError("Start date must be before end date.")

        # ERDDAP query components
        url_base = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.nc?"
        spatial_part = f"[({min(self.lats)}):1:({max(self.lats)})][({min(self.lons)}):1:({max(self.lons)})]"

        urls = []

        current_start = start_date.replace(
            hour=12, minute=0, second=0
        )  # match ERDDAP default time
        while current_start < end_date:
            # End of current month or end_date, whichever comes first
            next_month = (current_start + relativedelta(months=1)).replace(day=1)
            current_end = min(next_month - timedelta(days=1), end_date)

            # Build the time slice
            time_slice = f"[({current_start.strftime('%Y-%m-%dT%H:%M:%SZ')}):1:({current_end.strftime('%Y-%m-%dT%H:%M:%SZ')})]"
            query = f"{self.var}{quote(time_slice + spatial_part)}"
            urls.append(f"{url_base}{query}")

            current_start = next_month

        return urls

    def remove_part_files(self, main_dir: Path) -> None:
        """
        Remove part files from the main directory.
        This is useful for cleaning up temporary files after downloads.
        """
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

    def get_output_fp(self, url: str) -> Path:
        """Generate the output file path for a given URL."""
        import re

        matches = re.findall(r"(\d{4}-\d{2}-\d{2})T", url)
        if matches:
            date_str = "_".join(matches)
        else:
            date_str = "unknown"

        return self.output_dir / f"{self.var}_{date_str}.nc"

    def file_exists(self, url: str) -> bool:
        """Check if a file already exists based on its URL."""
        file_path = self.get_output_fp(url)
        exists = file_path.exists() and file_path.stat().st_size > 0
        if not exists:
            # Clean up any leftover .part files
            part_file = file_path.with_suffix(file_path.suffix + ".part")
            if part_file.exists():
                try:
                    part_file.unlink()
                except Exception:
                    pass
        return exists

    def prune_urls(self, urls: list[str]) -> list[str]:
        """Prune URLs that already exist in the output directory."""
        return [url for url in urls if not self.file_exists(url)]

    def _process_url_with_ui(self, url, ui_instance):
        """Process a single URL with the UI."""
        try:
            file_id = url  # Use url as unique identifier
            # filename = self.get_output_fp(url).name
            ui_instance.set_status(file_id, "STARTING", "cyan")
            success = self._download_url_direct_ui(url, ui_instance)
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

    def _download_url_direct_ui(self, url, ui_instance):
        """Directly download a file from a URL using the UI."""
        file_path = self.get_output_fp(url)
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
            # Set the per-file progress bar total to the file size (if available)
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
        from rich.console import Console

        console = Console()
        self.remove_part_files(self.output_dir)
        urls = self.get_erdapp_thredds_nc_file_urls()

        urls_to_download = [u for u in urls if not self.file_exists(u)]
        if not urls_to_download:
            console.print("All files already exist, nothing to download.")
            return

        dl_str = "file" if len(urls_to_download) == 1 else "files"

        console.print(f"Downloading {len(urls_to_download)} new {dl_str}...")

        with ui.DownloadProgressUI(urls_to_download) as ui_instance:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                _ = [
                    executor.submit(self._process_url_with_ui, url, ui_instance)
                    for url in urls_to_download
                ]
            ui_instance.print_summary()
        end_time = time.localtime()
        console.print(f"\n:clock3: END: {time.strftime('%Y-%m-%d %H:%M:%S', end_time)}")


def main():
    erdapp = GetERDAPP()
    erdapp.run()


if __name__ == "__main__":
    main()
