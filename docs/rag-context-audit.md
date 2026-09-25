# Аудит потерь контекста в RAG-пайплайне Telegram-бота

Статус: рабочий список находок для поэтапного исправления.
Дата аудита: 2026-09-22.
Симптом, с которого начали: бот «плохо считывает входящие обращения, теряет
информацию, не до конца понимает контекст вопроса и плохо ищет информацию».

## Пайплайн коротко

```text
Telegram message
  -> bot/handlers/text.py            (берётся только message.text)
  -> POST /chats/{id}/messages       (content=str, optional media)
  -> app/chat/routes.py              (validate_input, moderation)
  -> ChatService.send_message        (history = последние 10 сообщений)
  -> RAGService.answer(question, history)
       _retrieval_plan               (LLM condense -> follow-up эвристика)
       dense retrieve x N запросов   (до 30 raw кандидатов)
       product filter (gts_nx)
       lexical full-text recall      (Qdrant text index, snowball RU)
       RRF fusion -> lexical-evidence rank -> optional BGE rerank
       dedup -> top-5 фрагментов
       min_score gate (0.45)
       _generate_answer              (контекст + сырой вопрос; истории НЕТ)
       citation grounding            (нет [N] -> ответ без списка источников)
  -> SSE -> bot stream_to_chat -> editMessageText
```

## Легенда

- **P0** — прямая потеря информации или контекста; чинить первыми.
- **P1** — заметная деградация качества или ложные отказы.
- **P2** — архитектурные белые пятна и риски на будущее.
- **Статус:** `[ ]` не начато, `[~]` в работе, `[x]` сделано, `[-]` отклонено с причиной.

---

## P0 — критичные потери контекста

### F1. История диалога не доходит до генератора

- **Слой:** RAG / генерация
- **Файлы:** `app/services/rag.py:1246-1267` (`_generate_answer`), вызов из `app/chat/service.py:230-234`
- **Суть:** prompt генератора = `RAG_SYSTEM_PROMPT + фрагменты + сырой текущий
  вопрос`. История диалога передаётся в `answer()`, но используется только для
  построения retrieval-запроса. Для follow-up «А какие из них для плоской
  деформации?» LLM видит эллиптический вопрос без единого сообщения истории —
  подстановка предмета происходит только в поиске, генератор о ней не знает.
- **Эффект:** главная причина «не понимает контекст вопроса».
- **Вариант исправления:** передавать в генерацию либо condensed-вопрос
  (`plan.ranking_question`), либо блок истории диалога; минимум — заменять
  сырой вопрос на расширенный для follow-up.
- **Статус:** [x] — в `_generate_answer` передаётся `history`; в prompt
  добавлен блок `Диалог:` (последние 6 сообщений, ≤600 символов каждое,
  `[N]`-цитаты прошлых ответов вырезаются). Тесты:
  `test_generation_prompt_includes_recent_dialog_without_stale_citations`,
  `test_generation_prompt_omits_dialog_section_without_history`,
  `test_generation_dialog_is_bounded_to_recent_turns`.

### F2. Медиа-контент выпадает из RAG-пути

- **Слой:** chat service -> RAG
- **Файлы:** `app/chat/service.py:230-234` (в `answer` уходит только
  `user_content`), `app/chat/media.py:64-92` (текст/транскрипт кладётся в
  `media_refs.part`), `bot/handlers/media.py` (content = заглушка)
- **Суть:** для фото, голоса, PDF/DOCX вопросом для retrieval и генерации
  становится заглушка `[фото]`, `[голосовое сообщение]`, `[документ x.pdf]`.
  Транскрипт whisper и текст документа используются только в модерации и в
  non-RAG ветке LLM.
- **Эффект:** любое обращение с вложением → поиск по заглушке → `NO_ANSWER`
  или случайные документы. Прямая потеря входящей информации.
- **Вариант исправления:** склеивать `media_refs.part.text` (для
  `type=="text"`) в вопрос для RAG; для `image_url` — отдельный путь
  (vision/caption) или явный отказ без поиска по заглушке.
- **Статус:** [x] — `send_message` строит вопрос для RAG через
  `_rag_question`: для `part.type=="text"` (транскрипт whisper, текст
  PDF/DOCX) вопрос = `user_content + part.text`, ограниченный
  `RAG_MEDIA_TEXT_MAX_CHARS` под лимит входа embedding-модели; для
  `image_url` с подписью — поиск по подписи, без подписи (заглушка
  `[фото]`) — явный отказ `MEDIA_IMAGE_NO_TEXT_REPLY` без retrieval.
  Тесты: `test_rag_question_merges_extracted_media_text`,
  `test_rag_question_bounds_extracted_document_text`,
  `test_rag_image_without_caption_refuses_without_retrieval`,
  `test_rag_image_with_caption_searches_caption_text`.

### F3. Валидатор входа отклоняет документы и длинные сообщения

- **Слой:** API boundary
- **Файлы:** `app/chat/routes.py:82-83`, `app/services/security/input_validator.py:9` (`MAX_INPUT_CHARS=4000`), `app/chat/media.py:13` (`MAX_DOCUMENT_CHARS=30_000`)
- **Суть:** извлечённый текст PDF/DOCX/голоса валидируется тем же лимитом
  4000 символов, что и пользовательский ввод. Любой реальный документ
  (>~4 КБ текста) → HTTP 400. Длинное сообщение пользователя (письмо клиента,
  стектрейс) → тоже 400.
- **Эффект:** загрузка документов фактически сломана; пользователь видит
  безликое «Не удалось обработать запрос. Проверьте данные...».
- **Вариант исправления:** не применять `validate_input` к extracted-тексту
  (это доверенный контент файла, а не prompt-инъекция), либо отдельный лимит;
  маппинг кодов ошибок в боте на понятные сообщения (связано с F21).
- **Статус:** [x] — extracted-текст (`part.type=="text"`) больше не проходит
  `validate_input`: он уже ограничен `MAX_DOCUMENT_CHARS` при извлечении
  (теперь и транскрипт whisper тоже) и проверяется модерацией по склеенному
  контенту. Пользовательский `content` валидируется с отдельным лимитом
  `MAX_MESSAGE_CHARS=16_000` (`validate_input(..., max_chars=...)`), при
  `rule=="length"` в 400-деталь добавлен `limit`. В `bot/_common.py`
  `backend_error_message` разводит 400 (length/injection), 403, 413, 415,
  422 (`media_processing_failed` — с серверным `message`), 429 и 5xx.
  Тесты: `test_chat_message_route_accepts_long_user_message`,
  `test_chat_message_route_rejects_overlong_user_message_with_limit`,
  `test_chat_message_route_accepts_long_extracted_document_text`,
  `test_chat_message_route_does_not_apply_input_rules_to_extracted_document_text`,
  `test_validate_input_honours_custom_length_limit`,
  `test_audio_transcript_is_bounded_to_document_limit`, маппинг ошибок —
  в `tests/bot/test_admin_feedback.py`.

### F4. Follow-up детекция узкая и одношаговая

- **Слой:** RAG / retrieval plan
- **Файлы:** `app/services/rag.py:1165-1226` (`_standalone_question`,
  `_is_context_dependent_followup`, `_ELLIPTICAL_FOLLOWUP_PREFIXES`,
  `_FOLLOWUP_REFERENCE_TERMS`)
- **Суть:** follow-up признаётся только при префиксе `а / и / тогда / что
  насчёт` или при анафоре в вопросе ≤4 токенов (либо wh-вопрос ≤12 токенов с
  ранней анафорой). «как» отсутствует в wh-сете: «как мне настроить этот
  параметр?» из 6 токенов уже не follow-up. Не покрыты «подробнее», «почему»,
  «а если», «например», уточнения без анафоры. В расширение мерджится только
  последний user-вопрос — ответ ассистента и ходы глубже одного шага
  теряются.
- **Эффект:** уточняющие вопросы уходят в поиск как самостоятельные → потеря
  предмета вопроса → промахи recall и `NO_ANSWER`.
- **Вариант исправления:** добавить «как» в wh-сет, расширить эллиптические
  префиксы, мерджить последний user + последний assistant, в перспективе —
  LLM-condense через `condense_hook` (см. F11).
- **Статус:** [x] — wh-сет вынесено в `_FOLLOWUP_QUESTION_WORDS` и расширено
  («как», «почему», «зачем», вся парадигма «какой»/«чей», EN wh-слова), порог
  позиции анафоры смягчён до ≤4; добавлены head-маркеры продолжения
  `_FOLLOWUP_HEAD_MARKERS` («но», «или», «если», «значит», «подробнее»,
  «например», «уточни» + EN), фразовые открытия `_ELLIPTICAL_FOLLOWUP_PHRASES`
  («то есть», «what about», «for example»), однословные маркеры
  `_FOLLOWUP_SINGLE_WORDS` («почему», «дальше», «ещё»…), короткие предложные
  фрагменты ≤4 токенов `_FOLLOWUP_FRAGMENT_PREPOSITIONS` («для 2D?», «у
  него?»), ранняя анафора (≤2 позиция) в вопросах ≤8 токенов; ведущие
  дискурсивные частицы срезаются `_FOLLOWUP_LEAD_PARTICLE_RE`;
  `_FOLLOWUP_REFERENCE_TERMS` дополнен предложными и косвенными формами
  («него», «неё», «ему», «ими», «тому», «этим»…). В `_standalone_question`
  расширение мерджит последний обмен — последний user-вопрос + последний
  ответ ассистента через `_followup_context_text` (цитаты `[N]` вырезаются,
  `NO_ANSWER` пропускается, ≤`FOLLOWUP_CONTEXT_CHARS`=400 символов на часть) —
  referent'ы из ответа ассистента и цепочки follow-up больше не теряются.
  Тесты: `test_how_question_with_early_reference_is_follow_up`,
  `test_widened_follow_up_markers_use_history_variant`,
  `test_follow_up_merges_last_assistant_answer_for_assistant_owned_referent`,
  `test_chained_follow_up_uses_latest_exchange_not_first_question`,
  `test_follow_up_merge_skips_abstention_assistant_answer`,
  `test_self_contained_why_question_without_reference_stays_independent`,
  `test_self_contained_question_with_late_possessive_stays_independent`.

### F5. Follow-up с lexical-hit получает ровно один фрагмент

- **Слой:** RAG / сборка контекста
- **Файлы:** `app/services/rag.py:542-547`
- **Суть:** `if plan.mode == "follow_up" and reserved: context_limit = 1` —
  на уточняющий вопрос генератор получает один фрагмент и ноль истории.
  Ошибочный lexical match → гарантированно неверный ответ.
- **Эффект:** хрупкость ровно в том сценарии, который чаще всего встречается
  в support-диалоге.
- **Вариант исправления:** убрать жёсткую единицу (минимум 2–3 фрагмента) или
  делать её условной на уверенность lexical match.
- **Статус:** [x] — `context_limit = 1` заменён на
  `min(self._context_limit(), FOLLOWUP_LEXICAL_CONTEXT_SOURCES=3)`:
  зарезервированный lexical-фрагмент остаётся первым в prompt, за ним идут
  до двух сильнейших кандидатов — при ошибочном lexical match генератор
  сохраняет корректирующий контекст вместо гарантированно неверного ответа.
  История диалога в генерацию уже доходит после F1.
  Тест: `test_follow_up_lexical_hit_keeps_dense_alternatives_in_context`.

---

## P1 — заметная деградация / ложные отказы

### F6. Bot-слой теряет часть входящей информации

- **Слой:** telegram bot
- **Файлы:** `bot/handlers/text.py:19-30`, `bot/handlers/media.py`
- **Суть:** отправляется только `message.text`. Теряются: `reply_to_message`
  (цитата клиентского обращения — типовой сценарий инженера), entities
  (`text_link` — URL пропадает), `edited_message` (правки игнорируются),
  медиагруппы (каждое фото альбома → отдельный запрос), склейка подряд
  идущих сообщений (вопрос, порезанный на 2–3 сообщения, уходит как
  независимые запросы).
- **Эффект:** потеря контекста обращения ещё до бэкенда.
- **Вариант исправления:** добавлять текст цитаты в content; использовать
  `html_text`/`md_text` для ссылок; обработчик edited_message; debounce-склейка
  burst-сообщений (300–800 мс); media_group handler.
- **Статус:** [x] — единый burst-буфер `bot/handlers/_burst.py` собирает
  подряд идущие сообщения чата (текст и медиа) в один backend-запрос:
  debounce 0.7 с для текста, 1.5 с если в пачке есть `media_group_id`,
  принудительный flush через 4 с и по `/clear` (`flush_pending`), лимит
  10 файлов. Альбомы скачиваются параллельно и уходят одним multipart
  запросом — `POST /chats/{id}/messages` принимает `media: list[UploadFile]`,
  `media_refs` несёт `parts` (и legacy `part` = первый), `ChatService`
  склеивает все text-части для модерации и RAG-вопроса, все image_url части
  доходят до multimodal-контекста, а в персисте сохраняются только
  text-части без base64. `reply_to_message` добавляется в content блоком
  `[сообщение, на которое отвечает пользователь]:` (≤1500 символов, для
  медиа-цитат — placeholder-описание). `text_link`-ссылки разворачиваются
  в `[label](url)` с UTF-16-корректными offset'ами (полный `md_text`
  отвергнут — он экранирует всю пунктуацию и мусорил бы в истории).
  `edited_message`: правка ещё не отправленного сообщения заменяет его в
  буфере; правка уже ушедшего уходит отдельным запросом; собственные правки
  бота (стриминг-редактирование ответов) отфильтрованы, правки в FSM-сценарии
  `/ask` игнорируются. Частичные ошибки вложений собираются и показываются
  одним сообщением. Тесты: `tests/bot/test_burst.py` (16 кейсов),
  `test_send_message_sends_every_media_file_under_one_field`,
  `test_chat_message_route_accepts_multiple_media_files`,
  `test_chat_message_route_rejects_second_failing_media_file`,
  `test_multi_part_media_all_parts_reach_provider_content`,
  `test_multi_part_persisted_refs_keep_text_but_drop_image_payloads`,
  `test_rag_question_merges_all_extracted_text_parts`,
  `test_rag_album_of_images_without_caption_still_refuses`,
  `test_moderation_content_covers_all_text_parts`.

### F7. Хрупкий контракт цитат: нет `[N]` → полный отказ

- **Слой:** RAG / grounding
- **Файлы:** `app/services/rag.py:1284-1317` (`_ground_answer`)
- **Суть:** если модель не дала ни одной валидной ссылки `[N]`, ответ
  целиком заменяется на `NO_ANSWER` — даже если он правильный, но ссылки
  оформлены иначе (`(1)`, «источник 1», `【1】`) или отсутствуют.
- **Эффект:** outcome `missing_valid_citation` в телеметрии; правильные
  ответы маскируются под «нет данных».
- **Вариант исправления:** fallback — вернуть ответ без источников или с
  мягкой пометкой, расширить распознавание форматов ссылок, логировать
  исходный ответ при отказе.
- **Статус:** [x] — `_ground_answer` нормализует альтернативные формы цитат
  к каноническому `[N]` до проверки: `【1】`/`【1†…】`, `[^1]`, словесные
  «источник 1»/«источники 1 и 2»/«source: 2»; mid-line `(1)` — только при
  полном отсутствии канонических маркеров (строчные `(1)`-перечисления
  остаются прозой). Ответ без валидных ссылок доставляется без списка
  источников: висячие `[N]` вне диапазона вырезаются, пишется warning
  `rag_answer_missing_citations` (длина и число источников, без текста),
  outcome `missing_valid_citation` сохранён. Полный отказ остаётся только
  для явной абстенции и ответа, сводящегося к висячим ссылкам.
  Тесты: `test_rag_delivers_uncited_answer_without_sources`,
  `test_rag_normalizes_cjk_citation_brackets`,
  `test_rag_normalizes_annotated_cjk_citation`,
  `test_rag_normalizes_footnote_citation`,
  `test_rag_normalizes_source_word_citations`,
  `test_rag_normalizes_multi_number_source_word_citations`,
  `test_rag_merges_mixed_citation_formats_into_one_numbering`,
  `test_rag_converts_midline_paren_citation_without_canonical_markers`,
  `test_rag_keeps_paren_numbers_when_canonical_citations_exist`,
  `test_rag_keeps_line_start_paren_enumeration_as_prose`,
  `test_rag_strips_dangling_citations_but_delivers_answer`,
  `test_rag_refuses_answer_that_is_only_dangling_citations`,
  `test_rag_uncited_llm_answer_is_delivered_without_sources`.

### F8. Тихая деградация при недоступном/несобранном RAG

- **Слой:** app startup / fallback
- **Файлы:** `app/main.py:63-74`, `app/main.py:122-128`,
  `app/chat/service.py:236-244`, `app/routers/health.py:49-61`
- **Суть:** при `rag_service is None` (сборка упала: Qdrant, коллекция,
  размерность векторов, ingest) сообщения уходят в голый LLM-стрим без
  system prompt и без базы знаний — `chat.system_prompt` у бот-чатов `None`.
  `/ready` проверяет только Redis — отказ RAG нигде не виден. Отдельно:
  `RAG_ALLOW_EXTERNAL_DATA_TRANSFER=false` при включённом RAG → RuntimeError
  на каждый вопрос → «Не удалось завершить ответ».
- **Эффект:** бот отвечает выдумками вместо документации, и это никак не
  отличить снаружи.
- **Вариант исправления:** surface `rag_startup_error` в `/ready` и в ответах
  бота («база знаний недоступна»); дефолтный system prompt для non-RAG пути.
- **Статус:** [x] — `/ready` теперь возвращает `rag: up|down|disabled` и
  `rag_error` (`200` + `status: degraded` при падении RAG, чтобы compose
  healthcheck не рестартовал контейнер). При `rag_enabled` без живого сервиса
  SSE-поток открывает `{"type": "notice", "code": "rag_unavailable"}`; бот
  сохраняет notice в `last_response_notice` и дописывает его в финальное
  сообщение, не засоряя историю. Non-RAG fallback получает
  `FALLBACK_SYSTEM_PROMPT`, когда у чата нет своего system prompt.
  `RAG_ALLOW_EXTERNAL_DATA_TRANSFER=false` при включённом RAG переводится в
  startup-деградацию `ExternalDataTransferNotAllowed` вместо RuntimeError на
  каждый вопрос. Тесты: `test_initialize_rag_refuses_external_transfer_without_optin`,
  `test_initialize_rag_builds_when_external_transfer_allowed`,
  `test_ready_surfaces_rag_startup_failure_as_degraded`,
  `test_ready_reports_rag_up_when_service_built`,
  `test_chat_message_stream_notices_when_rag_enabled_but_unavailable`,
  `test_chat_message_stream_passes_rag_service_without_notice`,
  `test_chat_message_stream_without_rag_has_no_notice_when_disabled`,
  `test_non_rag_fallback_injects_default_system_prompt_when_chat_has_none`,
  `test_non_rag_fallback_keeps_custom_system_prompt`,
  `test_send_message_captures_notice_and_resets_between_responses`,
  `test_stream_to_chat_appends_notice_to_final_message`,
  `test_stream_to_chat_skips_empty_notice`.

### F9. `top_score` считается после дедупликации

- **Слой:** RAG / confidence gate
- **Файлы:** `app/services/rag.py:548-557`
- **Суть:** `top_score` — максимум по отобранным (post-dedup, post-limit)
  источникам, а не по пулу кандидатов. Если топ-скоринговый кандидат выпал
  на дедупликации (near-duplicate зарезервированного lexical) → ложный
  `low_score` → `NO_ANSWER`.
- **Эффект:** редкие, но реальные ложные отказы при наличии хорошего
  кандидата.
- **Вариант исправления:** считать confidence по пулу до dedup/лимита.
- **Статус:** [x] — `top_score` теперь максимум по пулу `ranked` до
  `_deduplicate`/context-limit: выпадение топ-кандидата на дедупликации
  (near-duplicate) или урезании контекста больше не даёт ложный
  `low_score` → `NO_ANSWER`, т.к. его evidence выживает в оставшемся
  близнеце/пуле. Телеметрия `rag_retrieval_completed` дополнительно пишет
  `selected_top_score` — максимум по фактически отобранным источникам.
  Тесты: `test_top_score_uses_candidate_pool_before_deduplication`;
  обновлён `test_llamaindex_reranks_after_top_k_and_guards_before_llm`.

### F10. Overview-эвристика урезает контекст на не-обзорных вопросах

- **Слой:** RAG / сборка контекста
- **Файлы:** `app/services/rag.py:103-107`, `1357-1359`, `544-551`
- **Суть:** `startswith(_OVERVIEW_PREFIXES)` → максимум 3 фрагмента и 1 на
  файл. «Расскажи, как настроить фильтрацию сетки» — фокусный вопрос, но
  попадает под обзорное урезание.
- **Эффект:** нехватка контекста на части сфокусированных вопросов.
- **Вариант исправления:** сузить детекцию обзоров (требовать «обзор»/«что
  такое» без конкретного действия) или не резать контекст ниже 5.
- **Статус:** [x] — `_is_overview_question` нормализует вопрос до токенов
  (пунктуация больше не влияет на префикс: «расскажи, как …» и «расскажи
  как …» распознаются одинаково) и отклоняет обзорную трактовку при
  признаках конкретного действия: wh-слова `_FOCUSED_QUESTION_WORDS`
  («как», «почему», «где», «сколько», how/why/where…) и основы действий
  `_FOCUSED_QUESTION_STEMS` («настро», «установ», «созда», «ошибк»,
  configure/install/error…). «Расскажи, как настроить фильтрацию сетки»
  теперь идёт по focused-пути с полным контекстом. Отдельный лимит
  `MAX_OVERVIEW_CONTEXT_SOURCES=3` убран: обзорные вопросы делят общий
  `_context_limit()`, а `max_per_file` в `_deduplicate` стал мягким
  предпочтением — отложенные кандидаты добираются в свободные слоты
  (backfill), поэтому контекст не пустеет, даже если вся evidence лежит
  в одном файле. Обзорный режим сохраняет инструкцию генерации, приоритет
  разнообразия файлов, подавление lexical-резерва при уверенном dense и
  телеметрию `answer_mode`.
  Тесты: `test_overview_question_keeps_full_context_and_prefers_diverse_files`,
  `test_survey_opener_with_focused_action_keeps_full_context`,
  `test_overview_question_backfills_context_from_single_file`,
  `test_overview_question_prefers_diverse_files_then_backfills`,
  расширен `test_overview_question_detection`.

---

## P2 — архитектурные белые пятна

### F11. Нет LLM-based query rewriting / condense

- **Файлы:** `app/services/rag.py:219`, `787-810`, `1175-1178`; `app/main.py:66`
- **Суть:** `condense_hook` объявлен, но нигде не инжектится — сервис
  создаётся без него. Весь disambiguation на regex-эвристиках; нет
  multi-query decomposition, HyDE, уточняющих вопросов пользователю.
- **Статус:** [x] — `_initialize_rag` инжектит `build_llm_condense_hook(config)`
  при `RAG_CONDENSE_ENABLED` + `RAG_CONDENSE_LLM_ENABLED` (оба default true).
  Хук отдаёт модели ограниченный блок `Диалог:` (тот же рендер, что в
  генерации) + последнее сообщение и просит один самостоятельный вопрос
  одной строкой; успешный rewrite идёт единственным dense-запросом
  (`plan_mode="condense_hook"`), эхо вопроса помечается `independent`.
  Ошибка провайдера, таймаут (отдельный лимит 15 с) или нарушение контракта
  вывода (пусто / >500 символов) → `None` → прежний детерминированный
  `follow_up`-merge с двумя запросами: падение LLM не отключает
  эвристики. Prompt помечает диалог и вопрос как данные, не инструкции.
  Lexical-запросы и окна строятся по терминам `plan.ranking_question` для
  переписанных режимов (`condense_hook`, `caller_standalone`), поэтому
  strict full-text recall ищет по разрешённому предмету, а не по
  эллиптической формулировке. Осознанно вне scope: multi-query
  decomposition, HyDE и уточняющие вопросы пользователю — отдельные
  инкременты. Тесты: `test_llm_condense_hook_*` (7 кейсов),
  `test_lexical_query_uses_condensed_question_terms`,
  `test_rag_config_reads_condense_llm_setting`,
  `test_initialize_rag_*condense*` (3 кейса).

### F12. Reranker выключен по умолчанию; ручные веса ранжирования

- **Файлы:** `app/core/config.py:190-193`, `app/services/rag.py:1062-1099`
- **Суть:** `RAG_RERANKER_ENABLED=false` по умолчанию; ранжирование =
  `cosine + 0.7·term_cov + 0.3·pair_cov`, подобранное под eval-кейсы.
- **Статус:** [x] — решение зафиксировано: BGE-реранкер остаётся opt-in
  (`RAG_RERANKER_ENABLED=false` по умолчанию), а ручные веса выведены из
  магических констант в конфигурацию: `RAG_LEXICAL_TERM_WEIGHT` (default 0.7)
  и `RAG_LEXICAL_PAIR_WEIGHT` (default 0.3) в `Settings`, `RAGConfig` и
  `_rank_by_lexical_evidence` с валидацией диапазона [0,1]. `0`/`0` отключает
  lexical-буст и возвращает чистый dense-порядок — это даёт управляемый
  способ сравнивать веса на eval-прогонах без правки кода. Формула
  задокументирована в `.env.example` и `docs/rag.md`. Тесты:
  `test_rag_config_reads_lexical_rank_weights`,
  `test_rag_config_defaults_keep_eval_tuned_lexical_weights`,
  `test_rag_config_rejects_out_of_range_lexical_weights` (4 кейса),
  `test_lexical_rank_weights_zero_disables_lexical_boost`.

### F13. Корпус узкий: только документация GTS NX

- **Файлы:** `docs/data_inventory.md`, `data/`, `app/services/rag.py:760-777`
- **Суть:** корпус = 9 PDF + 60 HTML + CHM по GTS NX. Установка, лицензии,
  типовые ошибки вне мануалов, другие продукты → структурно `NO_ANSWER`.
  Фильтр `RAG_PRODUCT=gts_nx` также отсекает явно помеченные другие
  продукты при их будущем добавлении.
- **Статус:** [x] — кодовая часть: новый `RAG_PRODUCT_SCOPE`
  (`strict|fallback|all`, default `fallback`) в Settings → `RAGConfig`.
  `fallback` сохраняет GTS NX первичным пулом и только при провале gate
  (`top_score < min_score` и нет выжившего lexical-evidence) повторяет тот же
  отбор по уже полученному полному корпусу — без дополнительных
  retrieval-запросов; `all` снимает фильтр; `strict` — прежнее поведение.
  Отбор пула вынесен в `_select_context` (fusion → lexical-rank → rerank →
  reserve → dedup → gate), обе итерации идентичны. Телеметрия:
  `product_scope` (`primary|fallback_all|all`) +
  `cross_product_selected_count`. `RAG_PRODUCT` значения `all`/`any`/`none`
  отклоняются (`RESERVED_PRODUCT_TOKENS`) в `RAGConfig` и `IngestionConfig`,
  чтобы scope-токен не стал мёртвым product-тегом. `infer_product` дополнен
  маркером `soilworks`; system/condense-промпты обобщены до «продуктов
  MIDAS». Уже лежащий в корпусе «Руководство по расчетам midas Civil
  2020.pdf» (раньше индексировался, но был недостижим) теперь отвечает на
  Civil-вопросы. Контентная часть (установка, лицензии, типовые ошибки, FAQ,
  другие продукты) — не код: пробелы и порядок пополнения зафиксированы в
  `docs/data_inventory.md` → «Покрытие и пробелы корпуса» и сознательно
  отложены пользователем до следующей версии агента — текущая версия
  работает только с мануалами.
  Тесты: `test_rag_product_scope_fallback_reaches_other_product_when_primary_empty`,
  `test_rag_product_scope_strict_keeps_other_product_unreachable`,
  `test_rag_product_scope_fallback_prefers_primary_evidence`,
  `test_rag_product_scope_fallback_engages_when_primary_below_threshold`,
  `test_rag_product_scope_all_disables_product_filter`,
  `test_rag_config_rejects_scope_tokens_as_target_product`,
  `test_rag_config_rejects_unknown_product_scope`,
  `test_rag_config_reads_product_scope_setting`,
  `test_ingestion_config_rejects_scope_tokens_as_default_product`.

### F14. Retrieval использует только последний user-вопрос

- **Файлы:** `app/services/rag.py:1181-1197`, `app/chat/service.py:216-217`
- **Суть:** история (10 сообщений) передаётся, но для расширения запроса
  берётся одно последнее user-сообщение. Детали обращения из первого
  сообщения (версия, код ошибки, конфигурация) выпадают из запроса уже на
  3–4 ходе.
- **Статус:** [x] — закрыто на трёх слоях. LLM-condense теперь рендерит всё
  переданное окно истории: `_dialog_context_section` получил параметр
  `max_messages` (`CONDENSE_DIALOG_HISTORY_MESSAGES=10` для хука,
  генерация по-прежнему ограничена 6 сообщениями), поэтому открывающее
  сообщение обращения остаётся видимым для rewrite и на поздних ходах.
  Детерминированный merge дополнен bounded keyword-tail
  `_detail_carry_tokens`: из user-ходов старше последнего переносятся
  только идентификаторы (числа и dotted-версии длиной ≥2, latin-термины
  ≥3; стоп-слова, маркеры продолжения и анафоры исключены
  `_DETAIL_CARRY_EXCLUDED_TOKENS`), до `FOLLOWUP_DETAIL_CARRY_TOKENS=8`
  токенов и без повторения терминов, уже присутствующих в последнем
  обмене; кириллические топик-слова сознательно не переносятся, чтобы
  ушедшая тема не возвращалась в запрос. Для `follow_up` в
  `_lexical_queries` добавлен incident-anchored вариант строгого запроса
  (до 2 detail-токенов + до 2 терминов текущего ограничения) — он же
  становится одним из dense-fallback запросов при провале strict-пула.
  Тесты: `test_follow_up_merge_carries_incident_details_from_earlier_user_turns`,
  `test_follow_up_detail_carry_skips_cyrillic_topic_words`,
  `test_follow_up_detail_carry_skips_terms_already_in_latest_exchange`,
  `test_detail_carry_tokens_keep_versions_and_identifiers_only`,
  `test_follow_up_lexical_queries_add_incident_detail_anchor`,
  `test_llm_condense_hook_renders_full_history_window`.

### F15. Групповые чаты делят историю

- **Файлы:** `bot/handlers/_common.py:28-29`
- **Суть:** `owner_external_id = message.chat.id` — у группы одна общая
  история и контекст; follow-up одного человека склеивается с вопросом
  другого.
- **Статус:** [x] — owner-ключ теперь зависит от отправителя:
  `chat_owner_external_id`/`message_owner_external_id` в `_common.py` дают
  приватным чатам прежний `chat.id`, а участникам групп —
  `"<chat_id>:<sender_id>"` (sender = `from_user`, для анонимных админов и
  постов канала — `sender_chat`, без отправителя — fallback `chat_id:chat_id`).
  Каждый участник группы получает отдельный backend-чат и свою историю.
  Burst-батчи `_pending` ключированы owner-ключом — быстрые сообщения разных
  участников не склеиваются в один запрос; `flush_pending` по-прежнему
  принимает Telegram `chat_id` и сбрасывает все батчи этого чата. Ответы в
  группах уходят через `message.reply` (`send_answer`), поэтому видно, к чьему
  вопросу относится ответ; доставка остаётся в тот же Telegram-чат.
  Feedback-колбэки вычисляют owner по нажавшему пользователю
  (`callback_owner_external_id`) — чужой клик получает 404 от backend и тост
  об ошибке; `_principal_headers_for_chat` больше не перезаписывает
  `_chat_owners` caller-supplied owner, иначе чужой клик отравлял бы привязку
  чата до рестарта. Ведущий сегмент ключа парсится обратно в Telegram chat id
  через `telegram_chat_id_from_owner` (`app/chat/domain.py`) для
  admin-уведомлений (`/chats/admin/{id}/notification`) и рассылки; рассылка
  дедуплицирует участников одной группы — одно сообщение на Telegram-чат.
  Побочный эффект: `/clear` участника теперь очищает только его историю, а
  admin-статистика считает «пользователь в чате». Тесты:
  `test_private_message_owner_key_stays_chat_id`,
  `test_group_message_owner_key_scopes_sender`,
  `test_group_message_owner_key_falls_back_to_sender_chat`,
  `test_group_message_owner_key_falls_back_to_chat_without_sender`,
  `test_callback_owner_key_uses_clicker_in_groups`,
  `test_callback_owner_key_private_chat`,
  `test_group_members_get_separate_histories`,
  `test_group_edit_lands_in_owning_member_batch`,
  `test_group_flush_pending_delivers_every_member_batch`,
  `test_group_feedback_uses_clicker_scoped_owner`,
  `test_private_feedback_uses_chat_id_owner`,
  `test_explicit_owner_does_not_rebind_chat_owner`,
  `test_broadcast_chat_ids_dedupes_group_members`,
  `test_broadcast_chat_ids_skips_unparseable_owners`,
  `test_system_message_notification_resolves_group_owner_chat_id`,
  `test_system_message_notification_rejects_non_telegram_owner`.

### F16. Ложные срабатывания ENCODING_PATTERNS; непонятные ошибки

- **Файлы:** `app/services/security/input_validator.py:26-30`
- **Суть:** `[A-Za-z0-9+/]{48,}` и base64-паттерны могут заблокировать
  вставленные license-ключи, хэши, подписи ошибок. Причина блокировки
  пользователю не раскрывается.
- **Статус:** [x] — блокировка по виду токена заменена проверкой содержимого:
  `ENCODING_PATTERNS` и эвристики «base64/decode/atob» убраны; длинные
  непрозрачные токены (`≥20` символов алфавита base64/base64url) декодируются,
  и блок срабатывает только если декодированный текст совпадает с
  `INJECTION_PATTERNS` (новое правило `encoded_injection`) либо rot13 всего
  сообщения раскрывает injection-маркер. License-ключи, sha256-хэши, JWT и
  base64-подписи ошибок (декодируются в бинарные данные или безобидный текст)
  теперь проходят; упоминания слов «base64», «atob()» не блокируются.
  Чтобы закодированный вредоносный запрос не обходил модерацию как
  непрозрачный текст, `decoded_fragments()` возвращает читаемые декодированные
  фрагменты, и `/chats/{id}/messages` добавляет их в `moderation_content`
  перед `ensure_input_allowed`. Ошибки стали понятнее: API-маршруты
  (`app/routers/chat.py`, `app/routers/rag.py`) проксируют `result.reason`,
  а бот маппит правила на разные подсказки — `injection` («похоже на попытку
  изменить инструкции ассистента»), `encoded_injection` («закодированный
  фрагмент со служебными инструкциями — уберите или перешлите открытым
  текстом»), `encoding` («слишком много нечитаемых символов»), `length`
  (лимит символов). Тесты:
  `test_validate_input_blocks_base64_marker`,
  `test_validate_input_blocks_base64_encoded_injection`,
  `test_validate_input_blocks_rot13_encoded_injection`,
  `test_validate_input_allows_license_keys_hashes_and_tokens`,
  `test_validate_input_allows_encoding_keyword_mentions`,
  `test_decoded_fragments_return_readable_payloads`,
  `test_decoded_fragments_skip_binary_tokens`,
  `test_chat_message_route_moderates_decoded_fragments`,
  `test_injection_400_maps_to_instruction_hint`,
  `test_encoded_injection_400_maps_to_decoded_hint`,
  `test_encoding_400_maps_to_charset_hint`.

### F17. Lexical recall только русский; токены <3 символов отбрасываются

- **Файлы:** `app/services/rag.py:640-671`, `960-969`
- **Суть:** snowball-стеммер русский — англоязычные запросы по английским
  мануалам почти без lexical-помощи. Токены <3 символов («3D», «2D»)
  отбрасываются из lexical-терминов.
- **Статус:** [x] — `_LEXICAL_TOKEN_RE` расширен до Unicode-букв и цифр
  (`[^\W_]+`): акцентированная латиница и другие письменности больше не
  дробятся на шумовые куски. Короткие токены с цифрой («3D», «2D», «R2»,
  «E5») принимаются общей проверкой `_is_lexical_term` в `_fulltext_terms`
  и `_lexical_terms` — тот же критерий, что уже действовал в
  `_detail_carry_tokens`; короткие токены без цифры по-прежнему
  отбрасываются. `_light_stem` получил консервативную английскую ветку
  (plural `-s`/`-es`/`-ies`, gerund `-ing` для токенов ≥7): стеммер
  симметричен для запроса и документа, поэтому «boundaries» совпадает с
  «boundary» в `_rank_by_lexical_evidence`, `_rank_fulltext_evidence` и
  `_minimum_cover_char_span`, а «stress»/«analysis»/«status» не режутся.
  Порог достаточности вынесен в `_sufficient_lexical_terms`: ≥3 термина,
  пара с суммой ≥9 символов или с digit-идентификатором («3d модель»),
  либо одиночный идентификатор («4021») — без него короткие термины не
  доходили бы до строгого запроса. Qdrant-индекс не менялся: на поле
  возможен только один stemmer, корпус русский — RUSSIAN snowball
  сохранён, а латинские термины в `MatchText` идут exact-match, как и
  прежде. Тесты:
  `test_fulltext_terms_keep_digit_identifiers_and_accented_latin`,
  `test_fulltext_terms_still_drop_short_noise`,
  `test_light_stem_converges_english_word_forms` (10 кейсов),
  `test_light_stem_leaves_stable_english_terms` (6 кейсов),
  `test_light_stem_keeps_russian_stemming`,
  `test_lexical_query_accepts_digit_identifier_pair`,
  `test_lexical_query_accepts_single_digit_identifier`,
  `test_lexical_query_still_rejects_lone_short_term`,
  `test_lexical_ranking_prefers_digit_identifier_match`,
  `test_lexical_ranking_matches_english_plural_to_singular`.

### F18. RAG-ответ не стримится; общий request_timeout=30с

- **Файлы:** `app/chat/service.py`, `app/services/rag.py`,
  `app/core/config.py`, `app/chat/routes.py`, `app/moderation/service.py`,
  `bot/services/backend_client.py`, `bot/handlers/_common.py`
- **Суть:** `_answer_with_rag` ждёт полный ответ → «Готовлю ответ…» висит до
  конца. `LLM_REQUEST_TIMEOUT` применён и к retrieval-embedding, и к
  генерации — сложный запрос с fallback-вариантами может упереться в 30 с.
- **Статус:** [x] — `RAGService.stream_answer()` отдаёт `delta`-события из
  `stream_complete` во время генерации и финальное `final`-событие с
  grounded-ответом, `top_score` и цитированными источниками (тот же
  retrieval-gate и `_ground_answer`, что у `answer()`). В
  `ChatService.send_message` синхронный генератор работает в worker-потоке
  и проксирует дельты в SSE немедленно через asyncio-очередь; каждый
  накопленный префикс проходит keyword-скрин
  `ModerationService.check_output_fragment` (заблокированный фрагмент не
  доходит до клиента; финальная `check_output` и запись в репозиторий
  сохранены), а `StreamReplace` после grounding/модерации подменяет
  показанный черновик каноническим текстом. Событие `replace` проходит
  через SSE-роут и `BackendClient` в `stream_to_chat`. Таймауты разделены:
  `request_timeout` (30 с) остаётся на Qdrant/embedding/retrieval, новый
  `RAG_GENERATION_TIMEOUT` (default 120 с) — на генерацию (`LlamaOpenAI` и
  bare-metal клиент); condense-hook ограничен `LLM_CONDENSE_TIMEOUT_SECONDS`
  как прежде. RAG-сервисы без `stream_answer` получают прежний буферный
  ответ. Тесты: стриминг/grounding/low-score/cumulative-chunks и разделение
  таймаутов в `test_rag.py`; живые дельты, replace и mid-stream модерация в
  `test_service_context.py`; SSE `replace` в `test_routes.py`; парсинг в
  `test_backend_client.py`; перезапись драфта в `test_streaming.py`.

### F19. `chat.system_prompt` игнорируется в RAG-пути

- **Файлы:** `app/chat/service.py:219-243`, `app/services/rag.py:39-48`
- **Суть:** для RAG всегда хардкод `RAG_SYSTEM_PROMPT`; кастомный промпт
  чата не применяется ни в RAG, ни (по сути) в non-RAG fallback, где у
  бот-чатов он `None`.
- **Статус:** [x] — `RAGService.answer`/`stream_answer`/`evaluate_inputs`/
  `query` принимают `system_prompt`; `ChatService` передаёт
  `chat.system_prompt` в оба пути (`stream_answer` и buffered `answer`)
  через `_rag_call_kwargs`, причём kwarg уходит только когда промпт задан —
  duck-typed сервисы без этого параметра не ломаются на чатах без промпта.
  `_answer_prompt` рендерит блок «Дополнительные инструкции этого чата»
  сразу после `RAG_SYSTEM_PROMPT` с явной оговоркой о приоритете базовых
  правил — опора на контекст, цитаты `[N]` и дословный отказ остаются
  обязательными, поэтому `_ground_answer` и abstention-детекция продолжают
  работать. Тесты: состав промпта
  (`test_answer_prompt_includes_chat_system_prompt`,
  `test_answer_prompt_omits_chat_section_without_custom_prompt`,
  `test_stream_answer_prompt_includes_chat_system_prompt`,
  `test_query_forwards_system_prompt`) и forwarding из чата
  (`test_rag_receives_chat_system_prompt_on_buffered_path`,
  `test_rag_streaming_receives_chat_system_prompt`,
  `test_rag_without_chat_system_prompt_gets_none`).

### F20. Два параллельных векторных хранилища

- **Файлы:** `app/main.py:115-121`, `app/core/config.py:144-153`, `224-233`
- **Суть:** чат-RAG использует `RAG_COLLECTION=rag_eval_chunk_1024`, агентные
  роутеры — `QDRANT_COLLECTION=documents` через отдельный `VectorStore`.
  Разные коллекции и разные сервисы — риск рассинхронизации и путаницы.
- **Статус:** [x] — приложение больше не содержит второго store. Фактически у
  параллельного `app.state.vector_store` не было живых потребителей: единственный
  инжектор `VectorStoreDep` не использовался ни одним роутером, а агентный путь
  ищет по локальной stub-базе (`agent_naive_tools`), не по Qdrant. Lifespan не
  создаёт и не закрывает store: `ensure_collection()` больше не является
  скрытым Qdrant-запросом на каждый старт и не роняет приложение при
  `QDRANT_URL` без доступного Qdrant (включая `RAG_ENABLED=false`). Мёртвый DI
  `get_vector_store`/`VectorStoreDep` удалён из `app/deps/providers.py`;
  `qdrant_collection`/`embedding_dim` удалены из `Settings` и `compose.yaml` —
  приложение структурно не может описать вторую коллекцию. `VectorStore`
  остаётся автономным адаптером homework-скриптов (`load_to_qdrant.py`,
  `vector_store_experiments.py`), которые читают `QDRANT_COLLECTION`/
  `EMBEDDING_DIM` из env напрямую с прежними дефолтами `documents`/`1024`;
  синтетический корпус `support_documents.json` намеренно отделён от боевого
  `RAG_COLLECTION`. `docs/vector_store.md`, `homework_5_2/README.md` и
  `.env.example` отражают разделение.
  Тесты: `test_lifespan_does_not_build_parallel_vector_store`,
  `test_settings_do_not_expose_experiment_vector_store_fields`.

### F21. Безликий маппинг ошибок в боте

- **Файлы:** `bot/handlers/_common.py:32-50`
- **Суть:** 400 (слишком длинный ввод / injection marker / документ не
  прошёл валидацию), 413, 415, 422 → одно и то же «Не удалось обработать
  запрос. Проверьте данные и попробуйте ещё раз». Пользователь не понимает,
  что пошло не так.
- **Статус:** [x] — сделано вместе с F3: `backend_error_message` разводит
  400 (`security_input_blocked` с учётом `rule`/`limit`), 403
  `moderation_blocked`, 413 `media_too_large`, 415 `unsupported_media_type`,
  422 `media_processing_failed` (прокидывает серверный `message`), 429 и
  5xx; прочее — прежний общий ответ.
  Добивка F21: `BackendStreamError` больше не теряет пользовательское
  `message` SSE-события `error` (`chat_stream_failed` — прокидывается
  серверный текст, пустое `message` → фолбэк «Ответ прервался…»), а
  транспортные сбои разведены: `ConnectError`/`ConnectTimeout`/
  `RemoteProtocolError` → «Сервис недоступен», любой `TimeoutException`
  (`Read`/`Write`/`Pool`) → «Ответ занимает слишком долго» — раньше все они
  падали в «Проверьте данные и попробуйте ещё раз».
  Тесты: `test_stream_error_surfaces_server_message`,
  `test_stream_error_without_message_uses_fallback`,
  `test_connect_timeout_maps_to_unavailable`,
  `test_remote_protocol_error_maps_to_unavailable`,
  `test_write_timeout_maps_to_timeout_hint`,
  `test_pool_timeout_maps_to_timeout_hint`,
  `test_unmapped_400_uses_generic_fallback`.

### F22. Конкурентные сообщения могут перемешать историю

- **Файлы:** `app/chat/service.py:216-283`, `app/chat/repositories/pg_repo.py:105-115`
- **Суть:** два быстрых сообщения → два параллельных `send_message` читают
  одну историю и делают `append_turn` в произвольном порядке — порядок пар
  «вопрос-ответ» может инвертироваться.
- **Статус:** [ ]

### F23. JsonChatRepository теряет сообщения при конкурентной записи

- **Слой:** chat repository (локальное/файловое хранилище)
- **Файлы:** `app/chat/repositories/json_repo.py:22-43` (`_locked`),
  `tests/chat/test_repository_contract.py::test_independent_json_repositories_serialize_concurrent_appends`
- **Суть (две гонки в filelock 3.32 на одном объекте):**
  1. **Доминирующая — реентерабельность инстанса.** `AsyncFileLock` хранит
     `lock_file_fd`/`lock_counter` в общем контексте инстанса: вторая корутина
     на том же объекте видит уже занятый fd и входит без OS-лока. Доказано
     детерминированно: 6 задач на одном `AsyncFileLock` → `max_inside=6`.
     То есть один инстанс репозитория вообще не сериализовал свои записи —
     любые два быстрых сообщения пользователя гонялись за `messages.jsonl`
     (Windows `O_APPEND` = seek+write, не атомарен → перезапись строки).
  2. **Вторичная — unlink гонка на Windows.** При release lock-файл удаляется;
     ожидающий инстанс держит fd на удалённый inode, новый acquirer создаёт
     свежий файл — оба «внутри» лока одновременно.
- **Эффект:** прямая потеря пользовательских сообщений в файловом режиме
  репозитория — ровно симптом «бот теряет информацию». До исправления:
  до 14 из 40 сообщений при конкурентных append'ах, флаки-тест ~50/50.
  Postgres-репозиторий не затронут.
- **Исправление:** двухуровневая блокировка `_locked()`: `asyncio.Lock`
  (внутриинстансовая эксклюзия корутин) поверх `AsyncFileLock(...,
  preserve_lock_file=True)` (межинстансовая/межпроцессная, без unlink гонки).
  `pyproject.toml`: `filelock>=3.32.0` — параметр `preserve_lock_file`
  существует только в 3.32+.
- **Статус:** [x] — стресс-репродукция 20/20 чисто (было до 14 потерь за
  прогон), флаки-тест 8/8, `tests/chat` 31 passed, полный прогон зелёный.

---

## Что сделано хорошо (не ломать)

- Честный abstention при низком score + нормализация отказов и
  перенумерация цитат (`_ground_answer`, `_is_abstention_answer`).
- `snippet` (700 chars) используется только для UI/API, в генерацию идёт
  полный чанк — не ломать это разделение.
- Lexical recall через Qdrant text-index с bounded window-фолбэками,
  RRF-fusion вариантов, резервирование lexical match.
- Телеметрия `rag_retrieval_completed`: outcome `low_score` /
  `missing_valid_citation` / `grounded` + размеры пулов — по ней можно
  локализовать любой провал без хранения текста вопроса.
- Безопасный рендеринг Telegram HTML, retry только на connect-ошибках,
  модерация входа/выхода.

## Конфигурационные проверки перед правками

- [ ] `.env`: `RAG_ENABLED=true`, `RAG_ALLOW_EXTERNAL_DATA_TRANSFER=true`,
      `RAG_SCORE_THRESHOLD`, `RAG_RERANKER_ENABLED`, `RAG_INGEST_ON_START`.
- [ ] Лог `rag_startup_failed` при старте app (если есть — F8 активен).
- [ ] Распределение outcome в `rag_retrieval_completed`: доля `low_score` vs
      `missing_valid_citation` vs `grounded` — локализует главный провал.

## Порядок работ (предложение)

| # | ID | Что даёт | Охват правки |
|---|----|----------|--------------|
| 1 | F1 | История/condensed-вопрос в генерацию | `app/services/rag.py` |
| 2 | F2 | Медиа-текст в RAG-запрос | `app/chat/service.py` (+ `media.py`) |
| 3 | F3 | Валидация extracted-текста + понятные ошибки | `app/chat/routes.py`, `bot/_common.py` (F21 заодно) |
| 4 | F4 | Расширение follow-up детекции и мерджа | `app/services/rag.py` |
| 5 | F5 | Убрать `context_limit=1` | `app/services/rag.py` |
| 6 | F7 | Смягчить citation-отказ | `app/services/rag.py` |
| 7 | F6 | Reply-quote, entities, edits, burst-склейка | `bot/handlers/` |
| 8 | F8 | Surfacing RAG-недоступности | `app/main.py`, `health.py`, `bot/_common.py` |
| 9 | F9, F10 | Confidence по пулу; overview-детекция | `app/services/rag.py` |
| 10 | F11+ | LLM-condense, reranker, корпус и прочее | по решению |

## Журнал выполнения

| Дата | ID | Что сделано | Проверка |
|------|----|-------------|----------|
| 2026-09-22 | — | Аудит проведён, список зафиксирован | read-only анализ |
| 2026-09-23 | F4 | Расширена детекция follow-up (wh-сет +«как», новые маркеры, предложные фрагменты, ранняя анафора); расширение мерджит последний обмен user+assistant | `pytest tests/unit/test_rag.py` |
| 2026-09-22 | F1 | `_generate_answer` принимает `history`; prompt включает ограниченный блок `Диалог:` перед `Контекст:`; `[N]`-цитаты прошлых ответов вырезаются | 3 новых теста + весь `tests/unit/test_rag.py` и `tests/chat/test_service_context.py` зелёные (71); полный прогон `tests` — 317 passed |
| 2026-09-22 | infra | Починен предсуществующий `IndentationError` в `app/services/agent_react_worker.py` (`finally:` без отступа) — блокировал импорт `app.main` и все route-тесты | сборка тестов восстановлена |
| 2026-09-22 | F23 | `JsonChatRepository`: добавлен `asyncio.Lock` поверх `AsyncFileLock(preserve_lock_file=True)` в `_locked()`; `filelock>=3.32.0` в pyproject | стресс-репродукция 20/20 без потерь (было до 14/40), флаки-тест 8/8, полный прогон `tests` зелёный |
| 2026-09-23 | F2 | `send_message` строит вопрос для RAG через `_rag_question`: `part.type=="text"` склеивается в вопрос с bound'ом `RAG_MEDIA_TEXT_MAX_CHARS`; фото без подписи — явный отказ без retrieval, с подписью — поиск по подписи | 4 новых теста в `tests/chat/test_service_context.py`; полный прогон `tests` — 322 passed |
| 2026-09-23 | F3, F21 | `validate_input` больше не применяется к extracted-тексту медиа; лимит пользовательского сообщения поднят до `MAX_MESSAGE_CHARS=16_000`; транскрипт whisper ограничен `MAX_DOCUMENT_CHARS`; бот разводит 400/403/413/415/422/429/5xx по понятным сообщениям | 4 новых route-теста + тесты валидатора, медиа-лимита и маппинга ошибок; полный прогон `tests` — 334 passed, 9 skipped |
| 2026-09-23 | F5 | `context_limit=1` для follow_up+reserved заменён лимитом `FOLLOWUP_LEXICAL_CONTEXT_SOURCES=3`: резерв идёт первым, до двух dense-альтернатив сохраняются в prompt | новый тест + полный прогон `tests` в `.venv` (`uv sync --extra eval`, CPython 3.13.15) — 350 passed, 9 skipped |
| 2026-09-23 | F7 | `_ground_answer`: нормализация `【N】`/`[^N]`/«источник N»/mid-line `(N)` к `[N]`; ответ без валидных ссылок доставляется без списка источников вместо `NO_ANSWER`; warning `rag_answer_missing_citations` | 12 новых тестов grounding + обновлённый uncited-кейс; полный прогон `tests` — 386 passed, 9 skipped |
| 2026-09-23 | F8 | `/ready` возвращает `rag` + `rag_error` (200 + degraded, без restart loop); SSE `notice`/`rag_unavailable` при включённом, но неподнявшемся RAG; `FALLBACK_SYSTEM_PROMPT` для non-RAG пути; `RAG_ALLOW_EXTERNAL_DATA_TRANSFER=false` → startup-деградация `ExternalDataTransferNotAllowed`; бот показывает notice под ответом | 12 новых тестов (startup, ready, route notice, fallback prompt, bot notice) + обновлённые ready-assertions |
| 2026-09-23 | F9 | `top_score` считается по пулу `ranked` до `_deduplicate`/context-limit вместо максимума по отобранным источникам — выпадение топ-кандидата на дедупе/лимите больше не даёт ложный `NO_ANSWER`; в телеметрию добавлен `selected_top_score` | новый тест near-duplicate + обновлённый reranker-кейс; прогон `tests` |
| 2026-09-23 | F10 | `_is_overview_question`: токен-нормализация (пунктуация не влияет) + veto по wh-словам и основам действий — «расскажи, как настроить …» больше не попадает под обзор; лимит `MAX_OVERVIEW_CONTEXT_SOURCES=3` убран, `max_per_file` стал мягким с backfill — контекст не режется ниже общего лимита ни на одном типе вопроса | 3 новых теста + переписанный overview-кейс + расширенный параметризованный детектор; полный прогон `tests` — 412 passed, 9 skipped |
| 2026-09-23 | F11 | `build_llm_condense_hook` инжектится в `_initialize_rag` (флаг `RAG_CONDENSE_LLM_ENABLED`): LLM-rewrite хода по ограниченному диалогу → одиночный dense-запрос `condense_hook`, эхо → `independent`; отказ/таймаут/невалидный вывод → `None` → deterministic `follow_up` merge (dual query) — `_standalone_question` теперь возвращает источник rewrite, чтобы падение hook не ломало эвристики; lexical-запросы берут термины из `plan.ranking_question` для переписанных режимов | 11 новых тестов (hook, план, fallback, lexical, config, startup-инжекция); полный прогон `tests` — 426 passed, 9 skipped |
| 2026-09-23 | F12 | Реранкер остаётся opt-in выключенным; веса `0.7·term_cov + 0.3·pair_cov` выведены из констант в `RAG_LEXICAL_TERM_WEIGHT` / `RAG_LEXICAL_PAIR_WEIGHT` (Settings → RAGConfig → `_rank_by_lexical_evidence`), валидация [0,1], `0`/`0` отключает lexical-буст | 7 новых тестов (from_settings, дефолты, диапазон, отключение буста); полный прогон `tests` — 434 passed, 9 skipped |
| 2026-09-23 | F17 | `_LEXICAL_TOKEN_RE` → `[^\W_]+` (Unicode-буквы/цифры не дробятся); короткие токены с цифрой («3D», «4021») принимаются через `_is_lexical_term`; `_light_stem` +консервативные английские plural/gerund редукции (симметрично для запроса и документа); достаточность терминов в `_sufficient_lexical_terms` — пара с идентификатором или одиночный идентификатор; Qdrant-индекс без изменений (один stemmer на поле — RUSSIAN сохранён, латиница в `MatchText` идёт exact-match) | 10 новых тестов (25 кейсов) в `test_rag.py`; полный прогон `tests` — 501 passed, 9 skipped |
| 2026-09-23 | F13 | `RAG_PRODUCT_SCOPE` (`strict\|fallback\|all`, default `fallback`): GTS NX остаётся первичным пулом, при провале evidence-gate отбор повторяется по полному корпусу без доп. retrieval-запросов — уже лежащий Civil-мануал становится достижимым; отбор вынесен в `_select_context`; телеметрия `product_scope` + `cross_product_selected_count`; scope-токены `all`/`any`/`none` отклонены как `RAG_PRODUCT`/`default_product`; `infer_product` +`soilworks`; промпты обобщены до продуктов MIDAS; пробелы корпуса описаны в `data_inventory.md` | 8 новых тестов в `test_rag.py` (scope-режимы, fallback, reserved-токены) + параметризованный guard в `test_ingestion.py`; полный прогон `tests` — 447 passed, 9 skipped |
| 2026-09-23 | F14 | condense-hook рендерит всё окно истории (`_dialog_context_section(max_messages=10)`; генерация по-прежнему 6 сообщений); heuristic-мердж дополнен keyword-tail `_detail_carry_tokens` — идентификаторы из user-ходов старше последнего (числа/dotted-версии ≥2, latin ≥3), до 8 токенов, без кириллических топик-слов и дублей; для `follow_up` добавлен incident-anchored вариант строгого lexical-запроса (details + current terms), попадающий и в dense-fallback | 6 новых тестов в `test_rag.py`; полный прогон `tests` — 453 passed, 9 skipped |
| 2026-09-23 | F18 | `RAGService.stream_answer` (delta→final, общий `_prepare_answer`/`_answer_prompt` с `answer()`); `ChatService.send_message` стримит RAG-дельты живьём через thread→queue мост с инкрементальным `check_output_fragment` и SSE-событием `replace` для канонического текста; бот обрабатывает `replace` в `stream_to_chat`; `RAG_GENERATION_TIMEOUT` (120с) отделён от `request_timeout` (30с, retrieval/Qdrant/embedding) | 13 новых тестов (rag streaming/timeouts, service deltas+replace+модерация, route SSE, client parse, bot replace); полный прогон `tests` — 514 passed, 9 skipped |
| 2026-09-25 | F19 | `answer`/`stream_answer`/`evaluate_inputs`/`query` +`system_prompt`; `_answer_prompt` вставляет блок «Дополнительные инструкции этого чата» после `RAG_SYSTEM_PROMPT` с оговоркой о приоритете базовых правил (контекст, `[N]`-цитаты, дословный отказ); `ChatService._rag_call_kwargs` передаёт `chat.system_prompt` в `stream_answer` и buffered `answer` только когда промпт задан — сигнатура вызова для duck-typed сервисов без параметра не меняется | 7 новых тестов (состав промпта в `test_rag.py`, forwarding в `test_service_context.py`); полный прогон `tests` — 521 passed, 9 skipped |
| 2026-09-25 | F20 | Из app удалён параллельный store: lifespan больше не создаёт `app.state.vector_store` (убран скрытый `ensure_collection` на каждый старт и падение приложения при недоступном Qdrant даже с `RAG_ENABLED=false`); мёртвые `get_vector_store`/`VectorStoreDep` удалены из DI; `qdrant_collection`/`embedding_dim` убраны из `Settings` и `compose.yaml` — единственный vector store приложения живёт в `RAGService` (`RAG_COLLECTION`); homework-скрипты читают `QDRANT_COLLECTION`/`EMBEDDING_DIM` из env напрямую; `docs/vector_store.md`, `homework_5_2/README.md`, `.env.example` обновлены | 2 новых теста в `tests/test_app_fastapi.py`; прогон `tests` |
