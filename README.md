# cosmo

A minimal terminal chat agent, like Claude Code, powered by Anthropic.

## Install

```bash
cd /Users/ziz/projet/agent
uv tool install .
```

## Configure

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# optional: pick your model (default: claude-sonnet-4-5)
export COSMO_MODEL=claude-sonnet-4-5
```

## Run

```bash
cosmo
```

Type `exit`, `quit`, or press Ctrl-C to leave.

## Dev (without installing)

```bash
uv run cosmo
```
# cosmo
