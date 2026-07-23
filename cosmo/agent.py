"""A minimal terminal chat agent, like Claude Code, powered by Anthropic."""

from __future__ import annotations

import getpass
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
from rich.text import Text

MODEL = os.environ.get("COSMO_MODEL", "claude-sonnet-4-5")
MAX_TOKENS = 4096
CONFIG_FILE = Path.home() / ".config" / "cosmo" / "api_key"

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


def run_terminal(command: str) -> str:
    """Execute a shell command (after confirmation) and return its output."""
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

    # Collapse long output so the terminal stays tidy.
    console.print(
        Panel(
            Text(_collapse(output), style="grey70"),
            title="[dim]output[/]",
            subtitle="[dim]collapsed[/]" if output.count("\n") >= 12 else None,
            border_style="grey37",
            padding=(0, 1),
        )
    )
    return output


def _collapse(text: str, head: int = 8, tail: int = 3) -> str:
    """Shorten multi-line text to a head/tail preview with a hidden-line note."""
    lines = text.splitlines()
    if len(lines) <= head + tail + 1:
        return text
    hidden = len(lines) - head - tail
    return "\n".join(
        lines[:head] + [f"… {hidden} more lines …"] + lines[-tail:]
    )



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

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
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
    """Stream one assistant turn.

    While generating, show a compact, transient "thinking" line that collapses
    away when done. Then print the full reply once as a single clean panel.
    """
    text_so_far = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    ) as stream:
        # transient=True => this live preview is erased when the block exits.
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

    # Print the finished reply once, as a stable (non-repainting) panel.
    if text_so_far.strip():
        console.print(_assistant_panel(text_so_far))


    # Rebuild clean content blocks (model_dump adds extra fields the API rejects).
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
    art.append(f"model: {MODEL}\n", style="dim")
    art.append("type ", style="dim")
    art.append("exit", style="bold")
    art.append(" or press ", style="dim")
    art.append("Ctrl-C", style="bold")
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
