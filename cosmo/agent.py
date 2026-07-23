"""A minimal terminal chat agent, like Claude Code, powered by Anthropic."""

from __future__ import annotations

import getpass
import json
import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

CONFIG_DIR = Path.home() / ".config" / "cosmo"
CONFIG_FILE = CONFIG_DIR / "api_key"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
MAX_TOKENS = 4096

console = Console()

SYSTEM_PROMPT = (
    "You are cosmo, a helpful terminal coding assistant. "
    "You can run shell commands with the run_terminal tool to inspect and "
    "modify the user's system. Prefer safe, read-only commands unless the "
    "user asks otherwise, and briefly explain what you are doing."
)

TOOLS = [
    {
        "name": "run_terminal",
        "description": (
            "Run a shell command on the user's machine and return its output. "
            "Use this to read files, list directories, run scripts, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
        },
    }
]


class Settings:
    """Live, editable, persisted settings for cosmo."""

    DEFAULTS = {
        "model": "claude-opus-4-8",
        "verbose": False,
    }

    def __init__(self) -> None:
        data = dict(self.DEFAULTS)
        if SETTINGS_FILE.exists():
            try:
                data.update(json.loads(SETTINGS_FILE.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
        # Env vars override the stored file at startup.
        if "COSMO_MODEL" in os.environ:
            data["model"] = os.environ["COSMO_MODEL"]
        if "COSMO_VERBOSE" in os.environ:
            data["verbose"] = os.environ["COSMO_VERBOSE"].lower() in {"1", "true", "yes"}
        self.model: str = data["model"]
        self.verbose: bool = bool(data["verbose"])

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(
            json.dumps({"model": self.model, "verbose": self.verbose}, indent=2)
        )

    def as_table(self) -> Table:
        table = Table(title="cosmo settings", title_style="bold cyan", border_style="cyan")
        table.add_column("setting", style="bold")
        table.add_column("value", style="green")
        table.add_row("model", str(self.model))
        table.add_row("verbose", "on" if self.verbose else "off")
        return table


settings = Settings()

# Remembers the most recent command output for /last.
_last_output: str = ""


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"on", "1", "true", "yes", "y"}


def handle_command(line: str) -> bool:
    """Handle a /slash command. Return True if the line was a command."""
    global _last_output
    if not line.startswith("/"):
        return False

    parts = line[1:].split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:]

    if cmd in {"settings", "config"}:
        console.print(settings.as_table())

    elif cmd == "set":
        if len(args) < 2:
            console.print("[yellow]usage:[/] /set <model|verbose> <value>")
        else:
            key, value = args[0].lower(), " ".join(args[1:])
            if key == "verbose":
                settings.verbose = _truthy(value)
                console.print(f"[green]✓[/] verbose = {'on' if settings.verbose else 'off'}")
            elif key == "model":
                settings.model = value
                console.print(f"[green]✓[/] model = {value}")
            else:
                console.print(f"[yellow]unknown setting:[/] {key}")
                return True
            settings.save()

    elif cmd == "last":
        if _last_output:
            console.print(
                Panel(Text(_last_output, style="grey70"), title="[dim]last output[/]",
                      border_style="grey37", padding=(0, 1))
            )
        else:
            console.print("[dim]no command output yet[/]")

    elif cmd == "help":
        console.print(
            Panel(
                "[bold]/settings[/]           show current settings\n"
                "[bold]/set model <name>[/]   switch the model\n"
                "[bold]/set verbose on|off[/] show or hide command output\n"
                "[bold]/last[/]               reprint the last command output\n"
                "[bold]/help[/]               show this help\n"
                "[bold]exit[/]                quit cosmo",
                title="[bold cyan]commands[/]",
                border_style="cyan",
                padding=(0, 1),
            )
        )

    else:
        console.print(f"[yellow]unknown command:[/] /{cmd}  (try /help)")

    return True


def run_terminal(command: str) -> str:
    """Execute a shell command (after confirmation) and return its output."""
    global _last_output
    console.print(
        Panel(
            Syntax(command, "bash", theme="ansi_dark", word_wrap=True),
            title="[bold yellow]run_terminal[/]",
            border_style="yellow",
            padding=(0, 1),
        )
    )
    try:
        confirm = Prompt.ask(
            "  [bold]Run this command?[/]", choices=["y", "n"], default="n"
        )
    except (EOFError, KeyboardInterrupt):
        confirm = "n"
    if confirm != "y":
        console.print("  [dim]declined[/]\n")
        return "Command was declined by the user."

    with console.status("[yellow]running…[/]", spinner="dots"):
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    output = (result.stdout or "") + (result.stderr or "")
    output = output.strip() or f"(no output, exit code {result.returncode})"
    _last_output = output

    n_lines = output.count("\n") + 1
    if settings.verbose:
        console.print(
            Panel(
                Text(output, style="grey70"),
                title="[dim]output[/]",
                border_style="grey37",
                padding=(0, 1),
            )
        )
    else:
        console.print(
            f"  [grey50]▸ output · {n_lines} line(s) hidden "
            f"[dim](/set verbose on, or /last)[/][/]"
        )
    return output


def load_api_key() -> str:
    """Get the API key from env, a stored file, or by prompting the user."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    if CONFIG_FILE.exists():
        key = CONFIG_FILE.read_text().strip()
        if key:
            return key

    console.print("[yellow]No API key found.[/] Let's set one up.")
    key = getpass.getpass("Enter your Anthropic API key: ").strip()
    if not key:
        raise SystemExit("No API key provided.")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(key)
    CONFIG_FILE.chmod(0o600)
    console.print(f"[green]✓[/] Saved API key to [dim]{CONFIG_FILE}[/]\n")
    return key


def run_tool(name: str, args: dict) -> str:
    """Dispatch a tool call to its implementation."""
    if name == "run_terminal":
        return run_terminal(args["command"])
    return f"Unknown tool: {name}"


def stream_turn(client: Anthropic, messages: list[dict]) -> tuple[str, list[dict]]:
    """Stream one assistant turn, then print the reply once as a clean panel."""
    text_so_far = ""
    with client.messages.stream(
        model=settings.model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    ) as stream:
        with Live(console=console, refresh_per_second=12, transient=True) as live:
            for text in stream.text_stream:
                text_so_far += text
                preview = text_so_far.strip().splitlines()[-1:] or ["…"]
                live.update(
                    Text.assemble(
                        ("cosmo ", "bold green"),
                        ("✎ ", "green"),
                        (preview[-1][:90], "dim"),
                    )
                )
            if not text_so_far:
                live.update(Text("cosmo ⚙ using a tool…", style="dim green"))
        final = stream.get_final_message()

    if text_so_far.strip():
        console.print(_assistant_panel(text_so_far))

    blocks: list[dict] = []
    for block in final.content:
        if block.type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return final.stop_reason, blocks


def _assistant_panel(markdown_text: str) -> Panel:
    body = Markdown(markdown_text) if markdown_text else Text("")
    return Panel(
        body,
        title="[bold green]cosmo[/]",
        border_style="green",
        padding=(0, 1),
    )


def _banner() -> Panel:
    art = Text()
    art.append("cosmo", style="bold cyan")
    art.append("  ·  your terminal agent\n", style="cyan")
    art.append(f"model: {settings.model}\n", style="dim")
    art.append("type ", style="dim")
    art.append("/help", style="bold")
    art.append(" for commands · ", style="dim")
    art.append("exit", style="bold")
    art.append(" to quit", style="dim")
    return Panel(art, border_style="cyan", padding=(1, 2))


def main() -> None:
    client = Anthropic(api_key=load_api_key())
    messages: list[dict] = []

    console.print(_banner())

    while True:
        try:
            user_input = Prompt.ask("[bold blue]you[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]bye 👋[/]")
            return

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            console.print("[cyan]bye 👋[/]")
            return
        if handle_command(user_input):
            continue

        messages.append({"role": "user", "content": user_input})

        # Agentic loop: keep going while the model wants to use tools.
        while True:
            stop_reason, blocks = stream_turn(client, messages)
            messages.append({"role": "assistant", "content": blocks})

            if stop_reason != "tool_use":
                break

            tool_results = []
            for block in blocks:
                if block["type"] == "tool_use":
                    output = run_tool(block["name"], block["input"])
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": output,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    main()
