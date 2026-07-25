"""A minimal terminal chat agent, like Claude Code, powered by Anthropic."""

from __future__ import annotations

import base64
import getpass
import json
import mimetypes
import os
import subprocess
import uuid
from datetime import datetime
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
SESSIONS_DIR = CONFIG_DIR / "sessions"
MAX_TOKENS = 4096


console = Console()

SYSTEM_PROMPT = (
    "You are cosmo, a helpful terminal coding assistant. "
    "You can run shell commands with the run_terminal tool to inspect and "
    "modify the user's system. Prefer safe, read-only commands unless the "
    "user asks otherwise, and briefly explain what you are doing."

    "FRUGALITY"
    "When you work, please clean the environement clean"
    "when you are done, clean the mess it there is"
)

RUN_TERMINAL_TOOL = {
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


def build_tools() -> list[dict]:
    """Assemble the tool list for a request, honoring live settings."""
    tools: list[dict] = [RUN_TERMINAL_TOOL]
    if settings.web_search:
        tools.append(
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": settings.web_search_max_uses,
            }
        )
    return tools


class Settings:
    """Live, editable, persisted settings for cosmo."""

    DEFAULTS = {
        "model": "claude-opus-4-8",
        "verbose": False,
        "web_search": False,
        "web_search_max_uses": 5,
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
        if "COSMO_WEB_SEARCH" in os.environ:
            data["web_search"] = os.environ["COSMO_WEB_SEARCH"].lower() in {"1", "true", "yes"}
        self.model: str = data["model"]
        self.verbose: bool = bool(data["verbose"])
        self.web_search: bool = bool(data["web_search"])
        self.web_search_max_uses: int = int(data["web_search_max_uses"])

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "verbose": self.verbose,
                    "web_search": self.web_search,
                    "web_search_max_uses": self.web_search_max_uses,
                },
                indent=2,
            )
        )

    def as_table(self) -> Table:
        table = Table(title="cosmo settings", title_style="bold cyan", border_style="cyan")
        table.add_column("setting", style="bold")
        table.add_column("value", style="green")
        table.add_row("model", str(self.model))
        table.add_row("verbose", "on" if self.verbose else "off")
        table.add_row("web_search", "on" if self.web_search else "off")
        table.add_row("web_search_max_uses", str(self.web_search_max_uses))
        return table


settings = Settings()


class Session:
    """A saved conversation: full transcript persisted as one JSON file."""

    def __init__(self, sid: str, created: str, model: str,
                 messages: list[dict] | None = None, title: str = "") -> None:
        self.id = sid
        self.created = created
        self.model = model
        self.messages: list[dict] = messages or []
        self.title = title

    @classmethod
    def new(cls) -> "Session":
        now = datetime.now()
        sid = now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        return cls(sid=sid, created=now.isoformat(timespec="seconds"),
                   model=settings.model)

    @property
    def path(self) -> Path:
        return SESSIONS_DIR / f"{self.id}.json"

    def save(self) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        # Title = first user message, trimmed, for a friendly listing.
        if not self.title and self.messages:
            first = next((m for m in self.messages if m["role"] == "user"), None)
            if first and isinstance(first["content"], str):
                self.title = first["content"][:60]
        self.path.write_text(json.dumps({
            "id": self.id,
            "created": self.created,
            "model": self.model,
            "title": self.title,
            "messages": self.messages,
        }, indent=2))

    @classmethod
    def load(cls, sid: str) -> "Session | None":
        path = SESSIONS_DIR / f"{sid}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(sid=data["id"], created=data["created"],
                   model=data.get("model", settings.model),
                   messages=data.get("messages", []),
                   title=data.get("title", ""))

    @staticmethod
    def list_all() -> list[dict]:
        if not SESSIONS_DIR.exists():
            return []
        items = []
        for path in SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                items.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(items, key=lambda d: d.get("created", ""), reverse=True)


# The conversation currently being recorded.
session: Session = Session.new()

# Remembers the most recent command output for /last.
_last_output: str = ""



def _truthy(value: str) -> bool:
    return value.strip().lower() in {"on", "1", "true", "yes", "y"}


def handle_command(line: str, messages: list[dict]) -> bool:
    """Handle a /slash command. Return True if the line was a command."""
    global _last_output, session
    if not line.startswith("/"):
        return False

    parts = line[1:].split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:]

    if cmd in {"settings", "config"}:
        console.print(settings.as_table())

    elif cmd == "sessions":
        rows = Session.list_all()
        if not rows:
            console.print("[dim]no saved sessions yet[/]")
        else:
            table = Table(title="cosmo sessions", title_style="bold cyan",
                          border_style="cyan")
            table.add_column("id", style="bold")
            table.add_column("date", style="dim")
            table.add_column("msgs", justify="right")
            table.add_column("title", style="green")
            for r in rows[:30]:
                created = r.get("created", "").replace("T", " ")
                table.add_row(r.get("id", "?"), created,
                              str(len(r.get("messages", []))),
                              r.get("title", "") or "[dim]—[/]")
            console.print(table)
            console.print("[dim]resume with[/] /resume <id>")

    elif cmd == "resume":
        if not args:
            console.print("[yellow]usage:[/] /resume <id>")
        else:
            loaded = Session.load(args[0])
            if loaded is None:
                console.print(f"[yellow]no session with id[/] {args[0]}")
            else:
                session = loaded
                settings.model = loaded.model or settings.model
                messages.clear()
                messages.extend(loaded.messages)
                console.print(
                    f"[green]✓[/] resumed [bold]{loaded.id}[/] "
                    f"([dim]{len(messages)} messages[/])"
                )

    elif cmd == "session":
        if args and args[0].lower() == "new":
            session = Session.new()
            messages.clear()
            console.print(f"[green]✓[/] started new session [bold]{session.id}[/]")
        else:
            console.print(
                f"current session [bold]{session.id}[/] · "
                f"[dim]{len(messages)} messages[/]"
            )


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
            elif key == "web_search":
                settings.web_search = _truthy(value)
                console.print(
                    f"[green]✓[/] web_search = {'on' if settings.web_search else 'off'}"
                )
            elif key == "web_search_max_uses":
                try:
                    settings.web_search_max_uses = max(1, int(value))
                    console.print(
                        f"[green]✓[/] web_search_max_uses = {settings.web_search_max_uses}"
                    )
                except ValueError:
                    console.print(f"[yellow]not a number:[/] {value}")
                    return True
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
                "[bold]/set web_search on|off[/] enable Anthropic web search\n"
                "[bold]/set web_search_max_uses <n>[/] cap searches per turn\n"
                "[bold]/image <path> [prompt][/] send an image for cosmo to read\n"
                "[bold]/last[/]               reprint the last command output\n"
                "[bold]/sessions[/]           list saved sessions\n"
                "[bold]/resume <id>[/]        continue a saved session\n"
                "[bold]/session new[/]        start a fresh session\n"
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


def build_image_message(path: str, prompt: str) -> dict | None:
    """Build a multimodal user message from an image file, or None on error."""
    fpath = Path(path).expanduser()
    if not fpath.exists():
        console.print(f"[yellow]no file:[/] {path}")
        return None
    media_type = mimetypes.guess_type(str(fpath))[0] or "image/png"
    if not media_type.startswith("image/"):
        console.print(f"[yellow]not an image:[/] {path} ({media_type})")
        return None
    try:
        data = fpath.read_bytes()
    except OSError as exc:
        console.print(f"[yellow]could not read:[/] {exc}")
        return None
    console.print(
        f"  [grey50]▸ attached image · {fpath.name} "
        f"[dim]({media_type}, {len(data) // 1024} KB)[/][/]"
    )
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(data).decode(),
                },
            },
            {"type": "text", "text": prompt or "Describe this image."},
        ],
    }


def run_tool(name: str, args: dict) -> str:
    """Dispatch a tool call to its implementation."""
    if name == "run_terminal":
        return run_terminal(args["command"])
    return f"Unknown tool: {name}"


# Tools cosmo executes locally; everything else (e.g. web_search) is a
# server-side tool resolved by the Anthropic API, not by us.
LOCAL_TOOLS = {"run_terminal"}


def stream_turn(client: Anthropic, messages: list[dict]) -> tuple[str, list[dict]]:
    """Stream one assistant turn, then print the reply once as a clean panel."""
    text_so_far = ""
    with client.messages.stream(
        model=settings.model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=build_tools(),
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

    _print_web_search(final)

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
        else:
            # Server-side tool blocks (server_tool_use, web_search_tool_result,
            # etc.) are resolved by the API. Keep them verbatim so the
            # transcript stays valid for follow-up turns.
            blocks.append(block.model_dump())
    return final.stop_reason, blocks


def _print_web_search(final) -> None:
    """Show a compact note whenever the model used server-side web search."""
    for block in final.content:
        btype = getattr(block, "type", None)
        if btype == "server_tool_use" and getattr(block, "name", "") == "web_search":
            query = ""
            if isinstance(getattr(block, "input", None), dict):
                query = block.input.get("query", "")
            console.print(f'  [magenta]\U0001F50E web search[/] · [dim]"{query}"[/]')
        elif btype == "web_search_tool_result":
            content = getattr(block, "content", None)
            n = len(content) if isinstance(content, list) else 0
            console.print(f"  [grey50]\u25B8 {n} result(s)[/]")


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
    # The live transcript IS the session's message list, so saving is trivial.
    messages: list[dict] = session.messages

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
        # /image builds a multimodal message and then lets the model reply,
        # so it is handled here rather than as a plain slash command.
        if user_input.lower().startswith("/image"):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 2:
                console.print("[yellow]usage:[/] /image <path> [prompt]")
                continue
            path = parts[1]
            prompt = parts[2] if len(parts) > 2 else ""
            image_msg = build_image_message(path, prompt)
            if image_msg is None:
                continue
            messages.append(image_msg)
        elif handle_command(user_input, messages):
            # /resume or /session new may have swapped the session; re-bind.
            messages = session.messages
            continue
        else:
            messages.append({"role": "user", "content": user_input})

        # Agentic loop: keep going while the model wants to use tools.
        while True:
            stop_reason, blocks = stream_turn(client, messages)
            messages.append({"role": "assistant", "content": blocks})

            if stop_reason != "tool_use":
                break

            tool_results = []
            for block in blocks:
                if block.get("type") == "tool_use" and block.get("name") in LOCAL_TOOLS:
                    output = run_tool(block["name"], block["input"])
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": output,
                        }
                    )
            if not tool_results:
                # Only server-side tools ran this turn; nothing to send back.
                break
            messages.append({"role": "user", "content": tool_results})

        # Auto-save the session after each completed turn.
        session.messages = messages
        session.save()



if __name__ == "__main__":
    main()
