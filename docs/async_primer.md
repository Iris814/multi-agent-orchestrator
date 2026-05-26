# Async Primer (1 page) — read before Phase 2

The Claude Agent SDK is async-first. If you've been writing synchronous pandas code, async will feel weird at first. Here's the minimum you need.

## The mental model

**Synchronous code** = a single worker who finishes each task before starting the next.

**Async code** = a single worker who can pause one task while waiting (e.g., for an API response) and switch to another. *One worker, but no idle time.*

## The 4 keywords you'll actually use

### `async def` — defines an async function (a "coroutine")

```python
async def fetch_data():
    return [1, 2, 3]
```

### `await` — pause here until the awaited thing finishes, then resume

```python
async def main():
    data = await fetch_data()      # wait for fetch_data
    print(data)
```

You can ONLY use `await` inside an `async def` function (or a Jupyter cell).

### `asyncio.run(...)` — entry point from sync code into async code

```python
import asyncio
asyncio.run(main())
```

This is the bridge: regular Python kicks off the async world.

### `async for` — iterate over an async generator (e.g., streaming agent output)

```python
async for chunk in agent.stream():
    print(chunk)
```

## Jupyter quirk

Jupyter has its own event loop running. Two ways to handle it:

**Option 1: top-level `await`** (works in Jupyter, NOT in regular .py files)
```python
# In a notebook cell:
result = await agent.run("hello")    # works!
```

**Option 2: `nest_asyncio`** (fallback if Option 1 errors)
```python
import nest_asyncio
nest_asyncio.apply()
asyncio.run(main())
```

Use Option 1 in our notebooks. Fall back to Option 2 only if you hit `RuntimeError: This event loop is already running`.

## Common gotchas

| Gotcha | Symptom | Fix |
|---|---|---|
| Forgot `await` | Function returns a `Coroutine` object instead of a value | Add `await` |
| Mixing sync/async | "coroutine was never awaited" warning | Wrap in `asyncio.run()` |
| Nested `asyncio.run()` | "asyncio.run() cannot be called from a running event loop" | Use `await` directly, or `nest_asyncio` |

## Cheatsheet for our project

```python
# In an .ipynb cell — uses Jupyter's event loop
result = await orchestrator.ask("What were Q2 2011 trends?")

# In streamlit_app.py — sync function calling async
import asyncio
def handle_query(question):
    return asyncio.run(orchestrator.ask(question))

# In tests/ — pytest-asyncio handles it
import pytest
@pytest.mark.asyncio
async def test_orchestrator():
    result = await orchestrator.ask("test")
    assert result is not None
```

That's enough to get going. The SDK abstracts most of the hard parts.
