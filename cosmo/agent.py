"""A minimal terminal chat agent, like Claude Code, powered by Anthropic.

The user interface is built with Textual: a scrollable transcript, a prompt
input, a live status line, a per-session stats footer, and a modal dialog for
confirming shell commands. All the Rich renderables (Panel, Markdown, Table,
Syntax) render natively inside Textual Static widgets, so the visual style
straight over from the old Rich REPL.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, Static

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOGO_PATH = PROJECT_DIR / "cosmo.png"
CONFIG_DIR = Path.home() / ".config" / "cosmo"
CONFIG_FILE = CONFIG_DIR / "api_key"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
SESSIONS_DIR = CONFIG_DIR / "sessions"
MAX_TOKENS = 4096
PRICES_FILE = CONFIG_DIR / "prices.json"
CONTEXT_LIMIT_FALLBACK = 200_000

# Prices are USD per 1M tokens (MTok). Model IDs + context windows come live
# from the Anthropic API (client.models.list); only prices are maintained here.
# Verify current values at:
# https://platform.claude.com/docs/en/about-claude/pricing
DEFAULT_PRICES = {
    "claude-opus-4-20250514":    {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-20250514":  {"input": 3.0,  "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
}


def load_prices() -> dict:
    """Local price overlay: built-in defaults merged with prices.json."""
    prices = dict(DEFAULT_PRICES)
    if PRICES_FILE.exists():
        try:
            prices.update(json.loads(PRICES_FILE.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return prices


_models_cache: list | None = None


def get_models(client: "Anthropic", refresh: bool = False) -> list[dict]:
    """Live model catalog from Anthropic, merged with local prices. Cached."""
    global _models_cache
    if _models_cache is not None and not refresh:
        return _models_cache
    prices = load_prices()
    models: list[dict] = []
    try:
        for m in client.models.list(limit=100):
            info = prices.get(m.id, {})
            models.append(
                {
                    "id": m.id,
                    "label": getattr(m, "display_name", m.id),
                    "context": getattr(m, "max_input_tokens", None) or CONTEXT_LIMIT_FALLBACK,
                    "max_out": getattr(m, "max_tokens", None),
                    "input": info.get("input"),
                    "output": info.get("output"),
                    "created": str(getattr(m, "created_at", ""))[:10],
                }
            )
    except Exception:  # network / auth / SDK issues
        return _models_cache or []
    _models_cache = models
    return models


def model_meta(client: "Anthropic", model_id: str) -> dict:
    """Price/context info for one model id, with a safe fallback."""
    for m in get_models(client):
        if m["id"] == model_id:
            return m
    prices = load_prices().get(model_id, {})
    return {
        "id": model_id, "label": model_id, "context": CONTEXT_LIMIT_FALLBACK,
        "input": prices.get("input"), "output": prices.get("output"),
    }


def estimate_cost(client: "Anthropic", model_id: str, in_tok: int, out_tok: int) -> float | None:
    """Estimated USD cost, or None if this model has no known price."""
    meta = model_meta(client, model_id)
    if meta.get("input") is None or meta.get("output") is None:
        return None
    return (in_tok / 1e6) * meta["input"] + (out_tok / 1e6) * meta["output"]


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
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls = 0
        self.last_input_tokens = 0

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
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "messages": self.messages,
        }, indent=2))

    @classmethod
    def load(cls, sid: str) -> "Session | None":
        path = SESSIONS_DIR / f"{sid}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        obj = cls(sid=data["id"], created=data["created"],
                  model=data.get("model", settings.model),
                  messages=data.get("messages", []),
                  title=data.get("title", ""))
        obj.input_tokens = data.get("input_tokens", 0)
        obj.output_tokens = data.get("output_tokens", 0)
        obj.tool_calls = data.get("tool_calls", 0)
        return obj

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


def load_api_key() -> str:
    """Get the API key from env or a stored file (prompting handled by the UI)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if CONFIG_FILE.exists():
        key = CONFIG_FILE.read_text().strip()
        if key:
            return key
    return ""


def save_api_key(key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(key)
    CONFIG_FILE.chmod(0o600)


def build_image_message(path: str, prompt: str) -> tuple[dict | None, str]:
    """Build a multimodal user message from an image file.

    Returns (message, note). ``message`` is None on error and ``note`` is a
    short status string to show in the transcript.
    """
    fpath = Path(path).expanduser()
    if not fpath.exists():
        return None, f"[yellow]no file:[/] {path}"
    media_type = mimetypes.guess_type(str(fpath))[0] or "image/png"
    if not media_type.startswith("image/"):
        return None, f"[yellow]not an image:[/] {path} ({media_type})"
    try:
        data = fpath.read_bytes()
    except OSError as exc:
        return None, f"[yellow]could not read:[/] {exc}"
    note = (
        f"  [grey50]▸ attached image · {fpath.name} "
        f"[dim]({media_type}, {len(data) // 1024} KB)[/][/]"
    )
    message = {
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
    return message, note


# Tools cosmo executes locally; everything else (e.g. web_search) is a
# server-side tool resolved by the Anthropic API, not by us.
LOCAL_TOOLS = {"run_terminal"}


def _assistant_panel(markdown_text: str) -> Panel:
    body = Markdown(markdown_text) if markdown_text else Text("")
    return Panel(
        body,
        title="[bold green]cosmo[/]",
        border_style="green",
        padding=(0, 1),
    )


def _logo_renderable(height: int = 12):
    """Return the cosmo.png rendered as terminal pixels, or None if unavailable."""
    if not LOGO_PATH.exists():
        return None
    try:
        from rich_pixels import Pixels
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(LOGO_PATH) as img:
            img = img.convert("RGBA")
            # Scale to the requested cell height, keeping aspect ratio.
            # Terminal cells are ~2x taller than wide, so double the width.
            ratio = img.width / img.height
            new_h = max(1, height)
            new_w = max(1, int(new_h * ratio * 2))
            img = img.resize((new_w, new_h))
            return Pixels.from_image(img)
    except Exception:
        return None


def _banner() -> Panel:
    art = Text()
    art.append("cosmo", style="bold cyan")
    art.append("  ·  your terminal agent\n", style="cyan")
    art.append(f"model: {settings.model}\n", style="dim")
    art.append("type ", style="dim")
    art.append("/help", style="bold")
    art.append(" for commands · ", style="dim")
    art.append("Ctrl+C", style="bold")
    art.append(" to quit", style="dim")
    logo = _logo_renderable()
    if logo is not None:
        from rich.console import Group
        body = Group(logo, Text(""), art)
    else:
        body = art
    return Panel(body, border_style="cyan", padding=(1, 2))


class ConfirmScreen(ModalScreen[bool]):
    """A yes/no modal used to approve running a shell command."""

    BINDINGS = [
        Binding("y", "confirm(True)", "Run", show=True),
        Binding("n,escape", "confirm(False)", "Decline", show=True),
    ]

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(
                Panel(
                    Syntax(self.command, "bash", theme="ansi_dark", word_wrap=True),
                    title="[bold yellow]run_terminal[/]",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )
            yield Label("[bold]Run this command?[/]  [green]y[/] / [red]n[/]", id="confirm-hint")

    def action_confirm(self, value: bool) -> None:
        self.dismiss(value)


class CosmoApp(App):
    """The Textual UI for cosmo."""

    ALLOW_SELECT = True

    CSS = """
    Screen {
        layout: vertical;
    }
    #transcript {
        height: 1fr;
        padding: 0 1;
        background: $surface;
        scrollbar-size-vertical: 1;
    }
    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $panel;
    }
    #prompt {
        border: round $primary;
        margin: 0 1 1 1;
    }
    #footer-line {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    ConfirmScreen {
        align: center middle;
    }
    #confirm-box {
        width: 80%;
        max-width: 100;
        height: auto;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }
    #confirm-hint {
        margin-top: 1;
        content-align: center middle;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear", "Clear transcript"),
    ]

    def __init__(self, client: Anthropic) -> None:
        super().__init__()
        self.client = client
        # The live transcript IS the session's message list, so saving is trivial.
        self.messages: list[dict] = session.messages
        self._busy = False

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="transcript")
        yield Static("", id="status")
        yield Input(placeholder="Message cosmo…  (/help for commands)", id="prompt")
        yield Static("", id="footer-line")

    def on_mount(self) -> None:
        self.title = "cosmo"
        self.log_widget = self.query_one("#transcript", VerticalScroll)
        self.status = self.query_one("#status", Static)
        self.footer_line = self.query_one("#footer-line", Static)
        self.write(_banner())
        self._refresh_footer()
        self.query_one("#prompt", Input).focus()

    # ----------------------------------------------------------------- helpers
    def write(self, renderable) -> None:
        """Write anything Rich-renderable (or markup str) to the transcript.

        Each entry is mounted as a selectable Static so users can
        mouse-select and copy text (RichLog did not allow selection).
        """
        static = Static(renderable, markup=isinstance(renderable, str))
        self.log_widget.mount(static)
        self.log_widget.scroll_end(animate=False)

    def set_status(self, text: str) -> None:
        self.status.update(text)

    def _refresh_footer(self) -> None:
        self.footer_line.update(self._footer())

    def _footer(self) -> Text:
        tin, tout = session.input_tokens, session.output_tokens
        cost = estimate_cost(self.client, settings.model, tin, tout)
        meta = model_meta(self.client, settings.model)
        ctx_limit = meta.get("context") or CONTEXT_LIMIT_FALLBACK
        ctx_pct = (session.last_input_tokens / ctx_limit * 100) if ctx_limit else 0
        cost_str = f"${cost:.4f}" if cost is not None else "$--"
        parts = [
            (f" {session.id} ", "dim"),
            ("· ", "grey37"),
            (f"{len(session.messages)} msgs ", "cyan"),
            ("· ", "grey37"),
            (f"↑{tin/1000:.1f}k ↓{tout/1000:.1f}k tok ", "green"),
            ("· ", "grey37"),
            (f"{cost_str} ", "yellow"),
            ("· ", "grey37"),
            (f"ctx {ctx_pct:.0f}% ", "blue"),
            ("· ", "grey37"),
            (f"{session.tool_calls} tools", "magenta"),
        ]
        return Text.assemble(*parts)

    # --------------------------------------------------------------- actions
    def action_clear(self) -> None:
        self.log_widget.remove_children()
        self.write(_banner())

    # ------------------------------------------------------------- input flow
    @on(Input.Submitted, "#prompt")
    def _on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self._busy:
            return
        if text.lower() in {"exit", "quit"}:
            self.exit()
            return

        self.write(Text.assemble(("you ", "bold blue"), (text, "white")))

        # /image builds a multimodal message and then lets the model reply.
        if text.lower().startswith("/image"):
            parts = text.split(maxsplit=2)
            if len(parts) < 2:
                self.write("[yellow]usage:[/] /image <path> [prompt]")
                return
            path = parts[1]
            prompt = parts[2] if len(parts) > 2 else ""
            image_msg, note = build_image_message(path, prompt)
            self.write(note)
            if image_msg is None:
                return
            self.messages.append(image_msg)
        elif text.startswith("/"):
            if self.handle_command(text):
                return
        else:
            self.messages.append({"role": "user", "content": text})

        self.run_turn_loop()

    # ------------------------------------------------------------- commands
    def handle_command(self, line: str) -> bool:
        """Handle a /slash command. Return True if fully handled (no model call)."""
        global _last_output, session
        parts = line[1:].split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:]

        if cmd in {"settings", "config"}:
            self.write(settings.as_table())

        elif cmd == "sessions":
            rows = Session.list_all()
            if not rows:
                self.write("[dim]no saved sessions yet[/]")
            else:
                table = Table(title="cosmo sessions", title_style="bold cyan",
                              border_style="cyan")
                table.add_column("", width=2)
                table.add_column("id", style="bold")
                table.add_column("date", style="dim")
                table.add_column("msgs", justify="right")
                table.add_column("title", style="green")
                for r in rows[:30]:
                    created = r.get("created", "").replace("T", " ")
                    mark = "★" if r.get("id") == session.id else ""
                    table.add_row(mark, r.get("id", "?"), created,
                                  str(len(r.get("messages", []))),
                                  r.get("title", "") or "[dim]—[/]")
                self.write(table)
                self.write("[dim]resume with[/] /resume <id>")

        elif cmd == "resume":
            if not args:
                self.write("[yellow]usage:[/] /resume <id>")
            else:
                loaded = Session.load(args[0])
                if loaded is None:
                    self.write(f"[yellow]no session with id[/] {args[0]}")
                else:
                    session = loaded
                    settings.model = loaded.model or settings.model
                    self.messages = session.messages
                    self.write(
                        f"[green]✓[/] resumed [bold]{loaded.id}[/] "
                        f"([dim]{len(self.messages)} messages[/])"
                    )
                    self._refresh_footer()

        elif cmd == "session":
            if args and args[0].lower() == "new":
                session = Session.new()
                self.messages = session.messages
                self.write(f"[green]✓[/] started new session [bold]{session.id}[/]")
                self._refresh_footer()
            else:
                self.write(
                    f"current session [bold]{session.id}[/] · "
                    f"[dim]{len(self.messages)} messages[/]"
                )

        elif cmd == "models":
            models = get_models(self.client, refresh=("refresh" in args))
            if not models:
                self.write("[dim]no models returned[/]")
                return True
            table = Table(title="cosmo models", title_style="bold cyan",
                          border_style="cyan")
            table.add_column("", width=2)
            table.add_column("model id", style="bold")
            table.add_column("name", style="green")
            table.add_column("context", justify="right", style="dim")
            table.add_column("$in/M", justify="right", style="yellow")
            table.add_column("$out/M", justify="right", style="yellow")
            for m in models:
                mark = "★" if m["id"] == settings.model else ""
                pin = f"{m['input']:.2f}" if m["input"] is not None else "—"
                pout = f"{m['output']:.2f}" if m["output"] is not None else "—"
                table.add_row(mark, m["id"], m["label"],
                              f"{m['context']:,}", pin, pout)
            self.write(table)
            self.write("[dim]switch with[/] /set model <id> "
                       "[dim]· /models refresh to re-fetch[/]")

        elif cmd == "pricing":
            if args and args[0] == "refresh":
                get_models(self.client, refresh=True)
                self.write("[green]✓[/] reloaded models + prices")
            else:
                self.write("[yellow]usage:[/] /pricing refresh")

        elif cmd == "set":
            if len(args) < 2:
                self.write("[yellow]usage:[/] /set <model|verbose> <value>")
            else:
                key, value = args[0].lower(), " ".join(args[1:])
                if key == "verbose":
                    settings.verbose = _truthy(value)
                    self.write(f"[green]✓[/] verbose = {'on' if settings.verbose else 'off'}")
                elif key == "model":
                    known = {m["id"] for m in get_models(self.client)}
                    if known and value not in known:
                        self.write(
                            "[yellow]⚠ not in model list[/] "
                            "[dim]· /models to see options[/]"
                        )
                    settings.model = value
                    self.write(f"[green]✓[/] model = {value}")
                    self._refresh_footer()
                elif key == "web_search":
                    settings.web_search = _truthy(value)
                    self.write(
                        f"[green]✓[/] web_search = {'on' if settings.web_search else 'off'}"
                    )
                elif key == "web_search_max_uses":
                    try:
                        settings.web_search_max_uses = max(1, int(value))
                        self.write(
                            f"[green]✓[/] web_search_max_uses = {settings.web_search_max_uses}"
                        )
                    except ValueError:
                        self.write(f"[yellow]not a number:[/] {value}")
                        return True
                else:
                    self.write(f"[yellow]unknown setting:[/] {key}")
                    return True
                settings.save()

        elif cmd == "last":
            if _last_output:
                self.write(
                    Panel(Text(_last_output, style="grey70"), title="[dim]last output[/]",
                          border_style="grey37", padding=(0, 1))
                )
            else:
                self.write("[dim]no command output yet[/]")

        elif cmd in {"clear", "cls"}:
            self.action_clear()

        elif cmd == "help":
            self.write(
                Panel(
                    "[bold]/settings[/]           show current settings\n"
                    "[bold]/set model <name>[/]   switch the model\n"
                    "[bold]/models [refresh][/]   list models (live) with prices\n"
                    "[bold]/pricing refresh[/]    reload model + price catalog\n"
                    "[bold]/set verbose on|off[/] show or hide command output\n"
                    "[bold]/set web_search on|off[/] enable Anthropic web search\n"
                    "[bold]/set web_search_max_uses <n>[/] cap searches per turn\n"
                    "[bold]/image <path> [prompt][/] send an image for cosmo to read\n"
                    "[bold]/last[/]               reprint the last command output\n"
                    "[bold]/sessions[/]           list saved sessions\n"
                    "[bold]/resume <id>[/]        continue a saved session\n"
                    "[bold]/session new[/]        start a fresh session\n"
                    "[bold]/clear[/]              clear the transcript (Ctrl+L)\n"
                    "[bold]/help[/]               show this help\n"
                    "[bold]exit[/]                quit cosmo (Ctrl+C)",
                    title="[bold cyan]commands[/]",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )

        else:
            self.write(f"[yellow]unknown command:[/] /{cmd}  (try /help)")

        return True

    # -------------------------------------------------------- terminal tool
    async def run_terminal(self, command: str) -> str:
        """Confirm via modal, run the command, and return its output."""
        global _last_output
        approved = await self.push_screen_wait(ConfirmScreen(command))
        if not approved:
            self.write("  [dim]declined[/]")
            return "Command was declined by the user."

        self.set_status("[yellow]⚙ running…[/]")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = (result.stdout or "") + (result.stderr or "")
            output = output.strip() or f"(no output, exit code {result.returncode})"
        except subprocess.TimeoutExpired:
            output = "(command timed out after 120s)"
        finally:
            self.set_status("")

        _last_output = output
        n_lines = output.count("\n") + 1
        if settings.verbose:
            self.write(
                Panel(
                    Text(output, style="grey70"),
                    title="[dim]output[/]",
                    border_style="grey37",
                    padding=(0, 1),
                )
            )
        else:
            self.write(
                f"  [grey50]▸ output · {n_lines} line(s) hidden "
                f"[dim](/set verbose on, or /last)[/][/]"
            )
        return output

    async def run_tool(self, name: str, args: dict) -> str:
        if name == "run_terminal":
            return await self.run_terminal(args["command"])
        return f"Unknown tool: {name}"

    # ----------------------------------------------------------- model turn
    def run_turn_loop(self) -> None:
        """Kick off the agentic loop as a background worker."""
        self._turn_worker()

    @work(exclusive=True)
    async def _turn_worker(self) -> None:
        self._busy = True
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = True
        try:
            # Agentic loop: keep going while the model wants to use tools.
            while True:
                stop_reason, blocks = await self.stream_turn()
                self.messages.append({"role": "assistant", "content": blocks})

                if stop_reason != "tool_use":
                    break

                tool_results = []
                for block in blocks:
                    if block.get("type") == "tool_use" and block.get("name") in LOCAL_TOOLS:
                        output = await self.run_tool(block["name"], block["input"])
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
                self.messages.append({"role": "user", "content": tool_results})

            # Auto-save the session after each completed turn.
            session.messages = self.messages
            session.save()
            self._refresh_footer()
        except Exception as exc:  # keep the app alive on API/network errors
            self.write(f"[red]error:[/] {exc}")
        finally:
            self.set_status("")
            prompt.disabled = False
            prompt.focus()
            self._busy = False

    async def stream_turn(self) -> tuple[str, list[dict]]:
        """Stream one assistant turn, updating the status line as text arrives."""

        def _do_stream():
            text_so_far = ""
            with self.client.messages.stream(
                model=settings.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=build_tools(),
                messages=self.messages,
            ) as stream:
                for text in stream.text_stream:
                    text_so_far += text
                    preview = text_so_far.strip().splitlines()[-1:] or ["…"]
                    line = preview[-1][:90]
                    self.call_from_thread(
                        self.set_status,
                        f"[bold green]cosmo[/] [green]✎[/] [dim]{line}[/]",
                    )
                if not text_so_far:
                    self.call_from_thread(
                        self.set_status, "[dim green]cosmo ⚙ using a tool…[/]"
                    )
                final = stream.get_final_message()
            return text_so_far, final

        text_so_far, final = await asyncio.to_thread(_do_stream)

        usage = getattr(final, "usage", None)
        if usage is not None:
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            session.input_tokens += in_tok
            session.output_tokens += out_tok
            session.last_input_tokens = in_tok

        self._print_web_search(final)

        if text_so_far.strip():
            self.write(_assistant_panel(text_so_far))

        blocks: list[dict] = []
        for block in final.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                session.tool_calls += 1
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

    def _print_web_search(self, final) -> None:
        """Show a compact note whenever the model used server-side web search."""
        for block in final.content:
            btype = getattr(block, "type", None)
            if btype == "server_tool_use" and getattr(block, "name", "") == "web_search":
                query = ""
                if isinstance(getattr(block, "input", None), dict):
                    query = block.input.get("query", "")
                self.write(f'  [magenta]\U0001F50E web search[/] · [dim]"{query}"[/]')
            elif btype == "web_search_tool_result":
                content = getattr(block, "content", None)
                n = len(content) if isinstance(content, list) else 0
                self.write(f"  [grey50]\u25B8 {n} result(s)[/]")


def _ensure_api_key() -> str:
    """Get a usable API key, prompting on the plain terminal if needed."""
    import getpass

    key = load_api_key()
    if key:
        return key
    print("No API key found. Let's set one up.")
    key = getpass.getpass("Enter your Anthropic API key: ").strip()
    if not key:
        raise SystemExit("No API key provided.")
    save_api_key(key)
    print(f"✓ Saved API key to {CONFIG_FILE}\n")
    return key


def main() -> None:
    client = Anthropic(api_key=_ensure_api_key())
    CosmoApp(client).run()


if __name__ == "__main__":
    main()
