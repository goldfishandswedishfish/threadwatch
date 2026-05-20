# threadwatch

Run an agentic task against multiple LLM providers simultaneously and compare how each handles long-horizon, multi-tool reasoning — side by side, in real time.

```bash
threadwatch run "Research the top 5 B2B CRM tools and output a markdown comparison table" \
  --provider gpt4o,https://openrouter.ai/api/v1,openai/gpt-4o \
  --provider llama,https://openrouter.ai/api/v1,meta-llama/llama-3.3-70b-instruct
```

## What it traces

Per provider, per step:
- Tool calls made (name, arguments, response)
- Tokens in / tokens out
- Latency
- Running context size

## Pathology detection

- **Repetition** — same tool called with identical args more than once
- **Backtracking** — repeated tool sequences
- **Hallucinated tool calls** — agent calls a tool not in the provided list
- **Context bloat** — token count growing faster than task progress

## Install

```bash
pip install git+https://github.com/goldfishandswedishfish/threadwatch
```

Requires Python 3.9+. Uses `httpx` for all HTTP — no LLM SDK dependencies.

## API keys

threadwatch auto-detects [1Password CLI](https://developer.1password.com/docs/cli) and injects secrets automatically on first run. Copy the example env file, fill in your `op://` paths, and place it at `~/.threadwatch.env`:

```bash
cp .env.1password.example ~/.threadwatch.env
```

Or set keys manually:

```bash
export OPENROUTER_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
export GROQ_API_KEY=gsk-...
export TAVILY_API_KEY=tvly-...   # optional — enables web_search tool
```

Key resolution order per provider:
1. `THREADWATCH_<PROVIDER_NAME>_KEY`
2. Well-known env var for the base URL (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, etc.)

## Tools

On first run, threadwatch creates `~/.threadwatch/tools/` and installs a starter set automatically:

- **calculator** — safe math expression evaluator, no external dependencies
- **web_search** — web search via Tavily (requires `TAVILY_API_KEY`)

From there the directory is yours — delete tools you don't want, edit existing ones, or drop in new `.py` files and they're picked up on the next run.

### Writing a custom tool

Each tool file needs a `DEFINITION` dict (OpenAI function format) and an async function with the same name:

```python
DEFINITION = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "..."}
            },
            "required": ["input"],
        },
    },
}

async def my_tool(input: str) -> str:
    return f"result for {input}"
```

See `examples/tools/` for working examples.

## Any OpenAI-compatible endpoint works

```bash
threadwatch run "your task" \
  --provider groq,https://api.groq.com/openai/v1,llama-3.3-70b-versatile \
  --provider openai,https://api.openai.com/v1,gpt-4o \
  --provider local,http://localhost:11434/v1,mistral
```

## Output

- **Live terminal view** — Rich-powered, one panel per provider, updates in real time
- **Final comparison table** — steps, tokens, latency, pathologies, completion status
- **JSON trace** — full run written to `./traces/<timestamp>.json`
