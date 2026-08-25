# Repository Instructions

## Non-negotiable error transparency

Errors must never be swallowed, replaced with generic text, or reduced to an
exception class name. Operators must receive the complete actionable reason for
a failure through exceptions, durable job state, error history, CLI output, and
public API consumers.

- When translating an exception, preserve its message and relevant context. For
  example, use `f"Download failed ({type(exc).__name__}): {exc}"`, not only
  `f"Download failed ({type(exc).__name__})"`.
- When an HTTP service returns an error, preserve the status and its bounded
  response detail. Do not report only `HTTP 404` when the response explains why.
- Never use `except ...: pass`, return a success value after a failure, or catch
  an exception merely to discard it.
- A catch used for cleanup, rollback, retry classification, or conversion to a
  domain exception is allowed only when the original detail is preserved or the
  exception is immediately re-raised.
- Expected predicate probes may return `False` only when failure is genuinely
  equivalent to a negative result and no operator action depends on the reason.
- Error text must be sanitized before persistence or display. Redact credentials,
  bearer tokens, license keys, and signed URLs, but retain all non-secret detail.
- Add regression tests that assert actionable detail survives every error
  boundary and that secrets remain redacted.

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
