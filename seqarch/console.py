# Copyright 2026 Kilian Ludwig
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

CONSOLE_THEME = Theme({
    "header": "bold bright_cyan",
    "header.accent": "bold bright_magenta",
    "section": "bold cyan",
    "label": "dim",
    "value": "bold white",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "muted": "dim",
    "model": "bright_cyan",
    "task": "bright_yellow",
    "metric.good": "bold green",
    "metric.ok": "bold yellow",
    "metric.bad": "bold red",
    "key": "bright_magenta",
})

console = Console(theme=CONSOLE_THEME, force_terminal=bool(os.environ.get("FORCE_COLOR")))


def print_section(title: str) -> None:
    console.print()
    console.print(Panel(
        Text(title, style="section"),
        border_style="cyan",
        padding=(0, 2),
    ))


def print_status(message: str, style: str = "success") -> None:
    icons = {"success": "[green]✓[/green]", "warning": "[yellow]![/yellow]", "error": "[red]✗[/red]"}
    icon = icons.get(style, "")
    console.print(f"  {icon} {message}")


def create_table(title: str, *columns: str, **kwargs) -> Table:
    table = Table(
        title=title,
        title_style="bold cyan",
        border_style="dim",
        header_style="bold bright_white",
        row_styles=["", "dim"],
        padding=(0, 1),
        **kwargs,
    )
    for col in columns:
        table.add_column(col)
    return table


def format_count(n: int) -> str:
    return f"{n:,}"


def accuracy_style(acc: float) -> str:
    if acc >= 0.8:
        return "metric.good"
    if acc >= 0.5:
        return "metric.ok"
    return "metric.bad"
