# Async benchmark results

Mode: `demo`
Requests: `20`
Demo delay: `0.100s`

| Mode | Concurrency | Total seconds | Errors | Speedup vs sync |
| --- | ---: | ---: | ---: | ---: |
| sync sequential | 1 | 2.191 | 0 | 1.00x |
| async batch_chat | 1 | 2.212 | 0 | 0.99x |
| async batch_chat | 5 | 0.440 | 0 | 4.98x |
| async batch_chat | 10 | 0.222 | 0 | 9.86x |

Notes:
- Demo mode uses `time.sleep` only in the old sync baseline and `asyncio.sleep` in the async fake provider.
- Real mode uses provider settings from `.env`; rate limits can reduce async speedup.
- `AsyncLLMClient` keeps concurrency in the instance semaphore `self._sem`.
