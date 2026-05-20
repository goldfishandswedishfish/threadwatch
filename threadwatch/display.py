from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .trace import ProviderTrace


console = Console()

STATUS_DONE = "[bold green]done[/]"
STATUS_ERROR = "[bold red]error[/]"
STATUS_RUNNING = "[yellow]running…[/]"


def _provider_panel(trace: ProviderTrace) -> Panel:
    lines: list[str] = []

    if trace.error and not trace.completed:
        status = STATUS_ERROR
    elif trace.completed:
        status = STATUS_DONE
    else:
        status = STATUS_RUNNING

    lines.append(f"[dim]model:[/] {trace.model}")
    lines.append(f"[dim]status:[/] {status}  [dim]elapsed:[/] {trace.elapsed_s:.1f}s")
    lines.append(
        f"[dim]steps:[/] {len(trace.steps)}  "
        f"[dim]tokens in:[/] {trace.total_tokens_in:,}  "
        f"[dim]tokens out:[/] {trace.total_tokens_out:,}  "
        f"[dim]tool calls:[/] {trace.total_tool_calls}"
    )
    lines.append("")

    for step in trace.steps[-6:]:
        step_line = f"[dim]step {step.step_num}[/]  {step.latency_ms:.0f}ms"
        if step.tool_calls:
            calls = ", ".join(
                f"[cyan]{tc.name}[/]([dim]{_fmt_args(tc.arguments)}[/])"
                for tc in step.tool_calls
            )
            step_line += f"  → {calls}"
        elif step.content:
            preview = step.content[:80].replace("\n", " ")
            step_line += f"  [green]↩ {preview}…[/]" if len(step.content) > 80 else f"  [green]↩ {preview}[/]"
        lines.append(step_line)

    if trace.pathologies:
        lines.append("")
        lines.append("[bold yellow]pathologies:[/]")
        for p in trace.pathologies:
            lines.append(f"  [yellow]⚠ {p}[/]")

    if trace.error:
        lines.append(f"\n[red]error: {trace.error}[/]")

    return Panel(
        "\n".join(lines),
        title=f"[bold]{trace.provider_name}[/]",
        border_style="green" if trace.completed else ("red" if trace.error else "blue"),
        padding=(0, 1),
    )


def _fmt_args(args: dict) -> str:
    if not args:
        return ""
    first_val = next(iter(args.values()), "")
    s = str(first_val)
    return s[:40] + "…" if len(s) > 40 else s


class LiveDisplay:
    def __init__(self, providers: list[tuple[str, str, str]]):
        self._traces: dict[str, ProviderTrace] = {}
        self._providers = providers
        self._live: Live | None = None

    def _render(self) -> Any:
        if not self._traces:
            return Text("Waiting for providers to start…", style="dim")
        panels = [_provider_panel(t) for t in self._traces.values()]
        if len(panels) == 1:
            return panels[0]
        layout = Layout()
        cols = [Layout(p, name=name) for p, (name, _, _) in zip(panels, self._providers)]
        layout.split_row(*cols)
        return layout

    def update(self, trace: ProviderTrace) -> None:
        self._traces[trace.provider_name] = trace
        if self._live:
            self._live.update(self._render())

    async def refresh_loop(self, interval: float = 0.15) -> None:
        while True:
            if self._live:
                self._live.update(self._render())
            await asyncio.sleep(interval)

    def show_final(self, traces: list[ProviderTrace]) -> None:
        console.print()
        console.rule("[bold]Final Comparison[/]")
        console.print()

        table = Table(box=box.ROUNDED, show_lines=True)
        table.add_column("Metric", style="dim", no_wrap=True)
        for t in traces:
            table.add_column(f"[bold]{t.provider_name}[/]\n[dim]{t.model}[/]", justify="right")

        def row(label: str, *vals: str):
            table.add_row(label, *vals)

        row("Status", *[
            "[green]✓ completed[/]" if t.completed else
            f"[red]✗ {t.error or 'incomplete'}[/]"
            for t in traces
        ])
        row("Steps", *[str(len(t.steps)) for t in traces])
        row("Total tokens in", *[f"{t.total_tokens_in:,}" for t in traces])
        row("Total tokens out", *[f"{t.total_tokens_out:,}" for t in traces])
        row("Total tokens", *[f"{t.total_tokens_in + t.total_tokens_out:,}" for t in traces])
        row("Tool calls", *[str(t.total_tool_calls) for t in traces])
        row("Total latency", *[f"{t.total_latency_ms / 1000:.2f}s" for t in traces])
        row("Wall time", *[f"{t.elapsed_s:.2f}s" for t in traces])
        row("Pathologies", *[
            "\n".join(f"⚠ {p}" for p in t.pathologies) if t.pathologies else "[green]none[/]"
            for t in traces
        ])

        console.print(table)

        for t in traces:
            if t.final_response:
                console.print()
                console.rule(f"[bold]{t.provider_name}[/] final response")
                console.print(t.final_response)


def run_live(coro, providers: list[tuple[str, str, str]]) -> list[ProviderTrace]:
    """Run an async coroutine under a Rich Live display and return its result."""
    import asyncio

    display = LiveDisplay(providers)
    result: list[ProviderTrace] = []

    async def _inner():
        with Live(display._render(), console=console, refresh_per_second=8, vertical_overflow="visible") as live:
            display._live = live
            refresh_task = asyncio.create_task(display.refresh_loop())
            try:
                traces = await coro(display.update)
                result.extend(traces)
            finally:
                refresh_task.cancel()
                try:
                    await refresh_task
                except asyncio.CancelledError:
                    pass

    asyncio.run(_inner())
    return result
