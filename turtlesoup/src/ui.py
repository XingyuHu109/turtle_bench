from __future__ import annotations

import os
from typing import Optional


class _PlainConsole:
    def __init__(self):
        pass

    def rule(self, title: str = ""):
        line = "=" * 10
        print(f"{line} {title} {line}")

    def box(self, title: str, body: str):
        lines = body.splitlines() if body else []
        width = max([len(title) + 2] + [len(l) for l in lines] + [20])
        top = "+" + "-" * (width + 2) + "+"
        print(top)
        print(f"| {title.ljust(width)} |")
        print("+" + "=" * (width + 2) + "+")
        for l in lines:
            print(f"| {l.ljust(width)} |")
        print(top)

    def info(self, msg: str):
        print(f"[INFO] {msg}")

    def warn(self, msg: str):
        print(f"[WARN] {msg}")

    # Progress as simple counter
    def progress_start(self, total: int, description: str = ""):
        self._prog_total = max(1, int(total))
        self._prog_n = 0
        self.info(f"{description} 0/{self._prog_total}")

    def progress_advance(self, step: int = 1, description: Optional[str] = None):
        self._prog_n = min(self._prog_total, self._prog_n + step)
        desc = description or ""
        self.info(f"{desc} {self._prog_n}/{self._prog_total}")

    def progress_stop(self):
        pass


class _RichConsole:
    def __init__(self):
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
        self.Console = Console
        self.Panel = Panel
        self.Progress = Progress
        self.SpinnerColumn = SpinnerColumn
        self.BarColumn = BarColumn
        self.TextColumn = TextColumn
        self.TimeElapsedColumn = TimeElapsedColumn
        self.console = Console()
        self._progress = None
        self._task_id = None

    def rule(self, title: str = ""):
        self.console.rule(title)

    def box(self, title: str, body: str):
        self.console.print(self.Panel.fit(body or "", title=title))

    def info(self, msg: str):
        self.console.print(f"[bold cyan]INFO[/]: {msg}")

    def warn(self, msg: str):
        self.console.print(f"[bold yellow]WARN[/]: {msg}")

    def progress_start(self, total: int, description: str = ""):
        if self._progress is not None:
            self.progress_stop()
        self._progress = self.Progress(
            self.SpinnerColumn(),
            self.TextColumn("[progress.description]{task.description}"),
            self.BarColumn(),
            self.TextColumn("{task.completed}/{task.total}"),
            self.TimeElapsedColumn(),
        )
        self._progress.start()
        self._task_id = self._progress.add_task(description, total=max(1, int(total)))

    def progress_advance(self, step: int = 1, description: Optional[str] = None):
        if self._progress is None or self._task_id is None:
            return
        if description:
            self._progress.update(self._task_id, description=description)
        self._progress.advance(self._task_id, step)

    def progress_stop(self):
        if self._progress is not None:
            try:
                self._progress.stop()
            except Exception:
                pass
        self._progress = None
        self._task_id = None


def get_console():
    use_rich = True
    if os.getenv("NO_RICH"):
        use_rich = False
    try:
        if use_rich:
            return _RichConsole()
    except Exception:
        pass
    return _PlainConsole()

