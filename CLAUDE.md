# Repository Instructions

## Non-negotiable API design

This is a deliberately async Python library. Avoid ambiguity: there is one public
execution model, and it is asynchronous.

- Public operations that can perform network, filesystem, database, subprocess,
  cryptographic, or other potentially blocking work MUST be `async def` and must
  not block the event loop.
- Public callbacks and providers MUST be async-only and explicitly typed as
  `Callable[..., Awaitable[T]]`. Do not accept both synchronous and asynchronous
  callback variants.
- Do not add synchronous client facades, synchronous alternatives, automatic
  sync/async detection, or dual execution paths for convenience.
- Do not run network I/O in worker threads. Use native async transports; HTTP
  uses one reusable `httpx.AsyncClient` connection pool.
- Isolate unavoidable blocking work such as filesystem calls, SQLite, hashing,
  and Sigstore verification behind clearly named internal adapters using
  `asyncio.to_thread()` or an executor.
- Callback execution context must be explicit: callbacks are awaited on the
  event-loop thread. A callback that needs blocking work is responsible for
  explicitly offloading that work.
- Keep resource ownership explicit. Async clients must provide deterministic
  async cleanup (`aclose()` and/or async context management).
- Do not expose internal synchronous storage or transfer primitives as public
  library APIs.

When compatibility conflicts with this contract before the first stable release,
prefer the clean async contract. Do not retain deprecated aliases or compatibility
layers unless explicitly requested.
