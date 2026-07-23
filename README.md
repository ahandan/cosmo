# cosmo

A minimal terminal chat agent, like Claude Code, powered by Anthropic.

## Install

```bash
cd /Users/ziz/projet/agent
uv tool install .
```

## Run

```bash
cosmo
```

On first launch, cosmo asks for your Anthropic API key and stores it at
`~/.config/cosmo/api_key` (file permissions `600`). After that it starts
straight into the chat.

You can also provide the key via env vars, which take priority:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# optional: pick your model (default: claude-sonnet-4-5)
export COSMO_MODEL=claude-sonnet-4-5
```


Type `exit`, `quit`, or press Ctrl-C to leave.

## Live settings (in-chat commands)

Change settings on the fly without restarting — they persist to
`~/.config/cosmo/settings.json`:

- `/settings` — show current settings
- `/set model <name>` — switch the model live
- `/set verbose on|off` — show or collapse command output
- `/last` — reprint the last command's full output
- `/help` — list commands

## Tools


cosmo is agentic: the model can call tools and cosmo runs them, feeding the
results back until the task is done.

- **run_terminal** — runs a shell command on your machine. For safety, cosmo
  prints each command and asks `y/N` before executing it.


## Dev (without installing)

```bash
uv run cosmo
```

