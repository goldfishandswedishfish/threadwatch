from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer

# Locations we check for a 1Password env file, in order.
_OP_ENV_CANDIDATES = [
    Path.home() / ".threadwatch.env",
    Path.cwd() / ".env.1password",
    Path(__file__).parent.parent / ".env.1password",
]

_SENTINEL = "THREADWATCH_OP_INJECTED"


def _keys_present() -> bool:
    # Sentinel is set inside the env file so we know op run already fired.
    return bool(os.environ.get(_SENTINEL))


def _ensure_secrets() -> None:
    """If required env vars are missing and op + an env file are available, re-exec under op run."""
    if _keys_present():
        return

    op = shutil.which("op")
    if not op:
        return  # op not installed — let the command fail naturally with a clear error

    env_file = next((p for p in _OP_ENV_CANDIDATES if p.exists()), None)
    if not env_file:
        return  # no env file found — nothing to inject

    # Re-exec this exact invocation under `op run --env-file=<file> -- <argv>`
    os.execvp(op, [op, "run", f"--env-file={env_file}", "--"] + sys.argv)

app = typer.Typer(
    help="Compare LLM providers on long-horizon agentic tasks.",
    no_args_is_help=True,
)

# Shell config files to patch, in priority order.
# We write to every file that already exists so the user's active shell gets it.
_SHELL_CONFIGS = [
    "~/.zshrc",
    "~/.zprofile",
    "~/.bash_profile",
    "~/.bashrc",
]
_MARKER = "# added by threadwatch"


_STARTER_TOOLS = ["calculator.py", "web_search.py"]
_EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "tools"


def _ensure_tools() -> None:
    """Copy starter tools to ~/.threadwatch/tools/ on first run if the directory doesn't exist."""
    from .tools import TOOLS_DIR
    if TOOLS_DIR.exists():
        return
    TOOLS_DIR.mkdir(parents=True)
    for filename in _STARTER_TOOLS:
        src = _EXAMPLES_DIR / filename
        if src.exists():
            shutil.copy(src, TOOLS_DIR / filename)
    typer.echo(
        f"[threadwatch] Created {TOOLS_DIR} with starter tools: {', '.join(t.replace('.py', '') for t in _STARTER_TOOLS)}\n"
        f"  Add or remove tools by editing files in that directory."
    )


def _ensure_on_path() -> None:
    """If the directory containing this script isn't in PATH, add it to shell configs."""
    script_dir = Path(sys.argv[0]).resolve().parent
    path_dirs = [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep)]
    if script_dir in path_dirs:
        return

    export_line = f'export PATH="{script_dir}:$PATH"  {_MARKER}\n'
    patched: list[str] = []

    for config in _SHELL_CONFIGS:
        config_path = Path(config).expanduser()
        if not config_path.exists():
            continue
        text = config_path.read_text()
        if str(script_dir) in text:
            return  # already present in at least one file — don't touch anything
        config_path.write_text(text.rstrip("\n") + "\n" + export_line)
        patched.append(config)

    if not patched:
        # No existing config found — create ~/.zshrc (macOS default since Catalina)
        zshrc = Path("~/.zshrc").expanduser()
        zshrc.write_text(export_line)
        patched.append("~/.zshrc")

    typer.echo(
        f"[threadwatch] Added {script_dir} to PATH in: {', '.join(patched)}\n"
        f"  Run `source {patched[0]}` or open a new terminal to apply."
    )


def _parse_provider(value: str) -> tuple[str, str, str]:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise typer.BadParameter(
            f"Expected name,base_url,model — got: {value!r}",
            param_hint="--provider",
        )
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    _ensure_on_path()
    _ensure_secrets()
    _ensure_tools()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="The task to run against all providers")],
    provider: Annotated[
        list[str],
        typer.Option("--provider", "-p", help="name,base_url,model  (repeatable)"),
    ] = [],
    max_steps: Annotated[int, typer.Option(help="Max agentic loop steps per provider")] = 20,
    no_export: Annotated[bool, typer.Option("--no-export", help="Skip JSON trace export")] = False,
):
    """
    Run TASK against each --provider in parallel and show a side-by-side trace.

    \b
    Example:
      threadwatch run "Compare top CRM tools" \\
        --provider groq,https://api.groq.com/openai/v1,llama-3.3-70b \\
        --provider openai,https://api.openai.com/v1,gpt-4o
    """
    if not provider:
        typer.echo("Error: at least one --provider is required.", err=True)
        raise typer.Exit(1)

    providers = [_parse_provider(p) for p in provider]

    from .loop import run_providers
    from .display import LiveDisplay, run_live, console
    from .export import export_traces

    from .tools import TOOL_NAMES, TOOLS_DIR

    console.print(f"\n[bold]threadwatch[/]")
    console.print(f"  [dim]task:[/]      {task}")
    console.print(f"  [dim]providers:[/] {', '.join(name for name, _, _ in providers)}")
    console.print(f"  [dim]tools:[/]     {', '.join(sorted(TOOL_NAMES)) or '[yellow]none — add tools to ' + str(TOOLS_DIR) + '[/]'}")
    console.print()

    traces = run_live(
        lambda on_update: run_providers(task, providers, max_steps, on_update),
        providers,
    )

    display = LiveDisplay(providers)
    display.show_final(traces)

    if not no_export:
        path = export_traces(task, traces)
        console.print(f"\n[dim]Trace saved → {path}[/]")


@app.command("tools")
def list_tools():
    """List the built-in tools available to agents."""
    from .tools import TOOL_DEFINITIONS
    from rich.console import Console
    from rich.table import Table

    c = Console()
    t = Table(show_header=True)
    t.add_column("Tool")
    t.add_column("Description")
    for d in TOOL_DEFINITIONS:
        fn = d["function"]
        t.add_row(fn["name"], fn["description"])
    c.print(t)
