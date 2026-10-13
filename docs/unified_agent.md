# Единый Telegram agent и retrieval по мануалам

Реализовано 2026-10-13. Telegram использует один разговор без `/agent` toggle:
пользователь задаёт вопрос, получает grounded ответ, уточняет его, просит
черновик и при необходимости явно просит отправку. Перед внешним действием
бот показывает точный `chat_id` получателя и полный текст с кнопками
«Подтвердить» / «Отменить».

## Границы реализации

- `app/services/rag.py`: public `retrieve` переиспользует production retrieval
  pipeline (follow-up context, product filters, dense/lexical recall, rerank,
  coverage/confidence gate, deduplication). Возвращает evidence без answer
  generation, с timeout/context budgets и chunk/document/page identity.
- `app/services/agent_persistent.py`: production RAG tool инъецируется лениво;
  knowledge miss (`no_evidence`) отличается от outage (`unavailable`). Нет
  fallback на `data/support_documents.json`. Citation numbers стабильны между
  поисками одного turn; финальный ответ ссылается только на выбранные sources.
- `app/chat/agent.py`: graph adapter с приватным interface/owner/chat thread.
  Пустой graph восстанавливает bounded context из существующей chat history.
- `app/chat/service.py`: тот же turn lock, media extraction, moderation и
  сохранение assistant messages с sources. Approval result также сохраняется.
- `bot`: прежний chat transport/SSE и feedback; added approval callback,
  `/history`, recovery preview при `/start`, полный split длинного текста.

Новых runtime dependencies, model IDs, pricing или env variables нет.
`AGENT_CHECKPOINTER=sqlite` — durable local default; production может использовать
existing PostgreSQL saver. Memory saver подходит для offline tests/demo и
не сохраняет pending после перезапуска процесса.

## API

| Операция | Контракт |
|---|---|
| `POST /rag/retrieve` | Authenticated `RAGQuery`; `status`, `question`, `sources`, `contexts`, telemetry; нет `answer` |
| `POST /chats` (Telegram) | Resolve durable owner/interface chat; reuse existing history after bot restart |
| `POST /chats/{id}/messages` | Existing multipart/SSE; final sources и optional `approval` event |
| `GET /chats/{id}/approval` | Exact pending preview или `null` |
| `POST /chats/{id}/approvals/{approval_id}` | JSON `{"approved": true/false}`; canonical result, message ID, sources, optional next approval |
| `DELETE /chats/{id}/messages` | Clear history + agent checkpoint; old buttons become stale |

Approval event/GET содержит только `approval_id`, recipient `chat_id`, `text`.
Recipient и text нельзя переопределить в resume request. Ownership проверяется
до обращения к graph; другой участник группы получает 404. Под тем же chat lock
сравнивается точный draft ID и pending node; stale/repeated callback даёт 409.
Raw `/agent/stream` также требует `approval_id` для resume и не принимает новое
input при pending approval. Telegram не использует административную роль `full`.

Доставка использует сохранённый idempotency key конкретного draft. Graph после
approve/cancel/error завершает turn детерминированным observed tool result;
следующий модельный вызов для утверждения «отправлено» не нужен. Unknown delivery
не отмечается как `sent` и требует проверки у получателя перед новой попыткой.

## Контекст, безопасность и ограничения

История graph ограничена пятью human-turn groups, tool results — 24 записями.
Provider input дополнительно подчиняется `CHAT_CONTEXT_WINDOW` и configured
token budget; удаляются целые human/tool groups, а переполнение текущей группы
даёт явный ответ без LLM call. Default model имеет output cap
`CHAT_RESPONSE_TOKENS`. Production retrieval сохраняет не более пяти выбранных
fragments на вызов, агент ограничивает context 8 000 символами на поиск.

Raw image bytes не попадают в checkpoint; captions, voice transcripts и
extracted document text проходят прежний media path. Фото без описания
получает прежнюю просьбу добавить текст. Output и exact outgoing draft
проходят moderation. Тексты мануалов считаются недоверенными данными; внешние
действия требуют отдельного подтверждения. Финальные unknown citations
удаляются, sources и feedback сохраняются с конкретным assistant message.

Retrieval timeout не может остановить уже начатый синхронный provider read.
Deadline запрещает следующие retrieval вызовы, а worker остаётся tracked и
держит request-budget lease до фактического завершения, включая disconnect.

Chat repository и graph saver не образуют общую транзакцию. Авария после graph
completion, но до append assistant message, может оставить chat history
отстающей от checkpoint. Идемпотентность доставки обеспечивается существующим
notifier/bot guard, а не общей exactly-once транзакцией всех компонентов.
При неизвестном исходе следует проверить получателя; слепое создание нового
действия может отправить дубликат. Per-chat locks рассчитаны на один API worker;
распределённая сериализация между несколькими API-процессами не добавлена.

## Воспроизводимая offline проверка

Из `HelpDesk_Assistant/` в managed CPython 3.13:

```powershell
.venv/Scripts/python.exe -m pytest tests/chat/test_unified_agent.py tests/test_agent_persistent.py tests/bot tests/chat tests/unit/test_rag.py -q
```

`tests/chat/test_unified_agent.py` проводит полный сценарий через настоящий
FastAPI ASGI transport, `BackendClient`, SQLite checkpoints, scripted model и
production RAG selection с injected retriever. Telegram sender заменён
`AsyncMock`, реальная доставка отсутствует. Проверяются:

1. ordinary question → actual RAG evidence identities → grounded sources;
2. follow-up в том же разговоре и hydration прежней chat history;
3. prepare exact draft → restart bot/backend graph → recover same chat/draft;
4. чужой, stale и повторный callback → отсутствие эффекта;
5. Confirm / Cancel → canonical observed result и persisted message;
6. `/clear` → history/checkpoint deletion и invalidation прежнего approval;
7. real miss, filtered product и unavailable/error без synthetic fallback;
8. несколько поисков с устойчивыми citation numbers;
9. timeout/disconnect worker lease retention, small token budget;
10. media text без raw image payload, полный escaped long preview;
11. public retrieval API без answer generation и с 503 для unavailable.

## Результаты проверки 2026-10-13

- `.venv/Scripts/python.exe -X utf8 -m pytest tests -q` — PASS:
  **628 passed, 11 skipped** (PostgreSQL repository checks требуют
  `CHAT_TEST_DATABASE_URL`). Полный прогон включает исправления восстановления
  текущего сообщения из истории и объединения источников параллельных tools.
- После финального исправления буквального отображения Markdown в exact
  preview: `.venv/Scripts/python.exe -X utf8 -m pytest tests/chat/test_unified_agent.py tests/bot -q`
  — PASS: **125 passed**. Подтверждаемый текст сохраняет `**` и backticks;
  HTML-символы экранируются, полный текст не обрезается.
- Ruff по полному набору правил, сравнение с `HEAD` — PASS: **0 новых
  замечаний**. Общий запуск `ruff check --select F,I` по изменённым файлам
  сообщает два существовавших `I001` в imports `app/services/rag.py` и
  `tests/unit/test_rag.py`; этот baseline не очищался массовым рефакторингом.
- `git diff --check` — PASS.

На Windows сетевые ограничения песочницы блокировали создание локального
`asyncio.socketpair` до выполнения тестов. Проверки выполнены вне песочницы;
LLM и Telegram actions в сквозном тесте заменены заглушками.

Live LLM, Telegram delivery и новый Docker smoke для этого изменения не
запускались. Офлайн-тесты проверяют integration contracts и ограничения, но
не заменяют проверку качества выбранной runtime модели на реальных запросах.
