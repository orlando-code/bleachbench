# rich
import logging
from datetime import datetime

# custom
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from bleachbench.utils import config


class MBDownloadColumn(DownloadColumn):
    def render(self, task):
        completed_mb = task.completed / 1024 / 1024
        total_mb = task.total / 1024 / 1024 if task.total else None
        if total_mb:
            return TextColumn(f"{completed_mb:.2f} / {total_mb:.2f} MB").render(task)
        else:
            return TextColumn(f"{completed_mb:.2f} MB").render(task)


class DownloadProgressUI:
    """Manages rich progress bars for overall and per-file download, with status in the bar description."""

    def __init__(self, files):
        self.files = files
        self.progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            MBDownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            transient=True,
            expand=True,
            auto_refresh=True,
            refresh_per_second=10,
        )
        self.overall_task = None
        self.status_counts = {
            "done": 0,
            "skipped": 0,
            "failed": 0,
            "unknown": 0,
        }
        self.failed_files = []  # List of (file_id, error_message)
        self.file_task_ids = {}  # file_id -> task_id
        self.file_status = {}  # file_id -> status string
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger(f"download_errors_{id(self)}")
        self.logger.setLevel(logging.ERROR)

        # Create logs directory next to the current file
        log_dir = config.get_repo_root() / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"download_errors_{timestamp}.log"

        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Clear existing handlers and add the new one
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(handler)

    def __enter__(self):
        self.progress.__enter__()
        self.overall_task = self.progress.add_task(
            "[cyan]Overall\n", total=len(self.files)
        )
        # For large numbers of files, only show progress bars for active downloads
        # to avoid memory issues and UI clutter
        if len(self.files) > 100:
            # Only create progress bars as files become active
            for file_id in self.files:
                self.file_task_ids[file_id] = None  # Will be created when needed
                self.file_status[file_id] = "PENDING"
        else:
            # For smaller numbers, show all progress bars immediately
            for file_id in self.files:
                desc = f"[white][PENDING] {self._display_name(file_id)}"
                total = 1  # Unknown size by default
                task_id = self.progress.add_task(desc, total=total, visible=True)
                self.file_task_ids[file_id] = task_id
                self.file_status[file_id] = "PENDING"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def set_status(self, file_id, status, color=None):
        """Update the status for a file and update the progress bar description."""
        color = color or "white"
        desc = f"[{color}][{status}] {self._display_name(file_id)}"
        task_id = self.file_task_ids[file_id]
        
        # Create progress bar on-demand for large file sets
        if task_id is None and status in ["STARTING", "DOWNLOADING"]:
            task_id = self.progress.add_task(desc, total=1, visible=True)
            self.file_task_ids[file_id] = task_id
        
        if task_id is not None:
            self.progress.update(task_id, description=desc)
        
        self.file_status[file_id] = status
        if status == "DONE":
            self.status_counts["done"] += 1
        elif status == "SKIPPED":
            self.status_counts["skipped"] += 1
        elif status == "FAIL":
            self.status_counts["failed"] += 1

    def add_failed(self, file_id, msg, exc_info=None):
        self.failed_files.append((file_id, msg))
        self.logger.error(
            f"File: {self._display_name(file_id)} - {msg}", exc_info=exc_info
        )

    def complete_file(self, file_id):
        # Advance overall progress and mark file bar as complete
        task_id = self.file_task_ids[file_id]
        if task_id is not None:
            self.progress.update(task_id, completed=self.progress.tasks[task_id].total)
        if self.overall_task is not None:
            self.progress.advance(self.overall_task)

    def update_file_progress(self, file_id, completed, total=None):
        task_id = self.file_task_ids[file_id]
        if task_id is not None:
            if total is not None:
                self.progress.update(task_id, completed=completed, total=total)
            else:
                self.progress.update(task_id, completed=completed)

    def hide_file(self, file_id):
        task_id = self.file_task_ids.get(file_id)
        if task_id is not None:
            self.progress.update(task_id, visible=False)

    def print_summary(self):
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        # Ensure the progress bar is cleared before printing summary
        self.progress.stop()

        console = Console()
        table = Table(
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Status", style="bold")
        table.add_column("Count", style="bold")
        table.add_row("[green]Completed[/green]", str(self.status_counts["done"]))
        table.add_row("[yellow]Skipped[/yellow]", str(self.status_counts["skipped"]))
        table.add_row("[red]Failed[/red]", str(self.status_counts["failed"]))
        table.add_row(
            "[white]Cancelled[/white]",
            str(len(self.files) - sum(self.status_counts.values())),
        )
        console.print(
            Panel(table, title="[white]Download Summary", border_style="white")
        )
        if self.failed_files:
            file_id, msg = self.failed_files[0]
            console.print(
                Panel(
                    f"Example failed file: [bold]{self._display_name(file_id)}[/bold]\n[red]{msg}[/red]",
                    title="[red]Failure Example[/red]",
                    style="red",
                )
            )

    def _display_name(self, file_id):
        # If file_id is a URL, show the last part or a date string; if a path, show the filename
        import re
        from pathlib import Path
        file_id = Path(file_id)
        if isinstance(file_id, str):
            # Try to extract a date or filename from the URL
            match = re.search(r"(\d{4}-\d{2})", file_id)
            if match:
                return f"{match.group(1)}.nc"
        if isinstance(file_id, Path):
            # If it's a path, show the filename
            # if file_id.startswith("/") or file_id.startswith("."):
                # return Path(file_id).name
            return file_id
        # Otherwise, just show the last part after a slash
        return file_id.split("/")[-1][:40]
        # return str(file_id)
