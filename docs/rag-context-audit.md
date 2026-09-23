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
       _retrieval_plan               (follow-up эвристика)
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
- **Статус:** [ ]

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
- **Статус:** [ ]

### F10. Overview-эвристика урезает контекст на не-обзорных вопросах

- **Слой:** RAG / сборка контекста
- **Файлы:** `app/services/rag.py:103-107`, `1357-1359`, `544-551`
- **Суть:** `startswith(_OVERVIEW_PREFIXES)` → максимум 3 фрагмента и 1 на
  файл. «Расскажи, как настроить фильтрацию сетки» — фокусный вопрос, но
  попадает под обзорное урезание.
- **Эффект:** нехватка контекста на части сфокусированных вопросов.
- **Вариант исправления:** сузить детекцию обзоров (требовать «обзор»/«что
  такое» без конкретного действия) или не резать контекст ниже 5.
- **Статус:** [ ]

---

## P2 — архитектурные белые пятна

### F11. Нет LLM-based query rewriting / condense

- **Файлы:** `app/services/rag.py:219`, `787-810`, `1175-1178`; `app/main.py:66`
- **Суть:** `condense_hook` объявлен, но нигде не инжектится — сервис
  создаётся без него. Весь disambiguation на regex-эвристиках; нет
  multi-query decomposition, HyDE, уточняющих вопросов пользователю.
- **Статус:** [ ]

### F12. Reranker выключен по умолчанию; ручные веса ранжирования

- **Файлы:** `app/core/config.py:190-193`, `app/services/rag.py:1062-1099`
- **Суть:** `RAG_RERANKER_ENABLED=false` по умолчанию; ранжирование =
  `cosine + 0.7·term_cov + 0.3·pair_cov`, подобранное под eval-кейсы.
- **Статус:** [ ]

### F13. Корпус узкий: только документация GTS NX

- **Файлы:** `docs/data_inventory.md`, `data/`, `app/services/rag.py:760-777`
- **Суть:** корпус = 9 PDF + 60 HTML + CHM по GTS NX. Установка, лицензии,
  типовые ошибки вне мануалов, другие продукты → структурно `NO_ANSWER`.
  Фильтр `RAG_PRODUCT=gts_nx` также отсекает явно помеченные другие
  продукты при их будущем добавлении.
- **Статус:** [ ]

### F14. Retrieval использует только последний user-вопрос

- **Файлы:** `app/services/rag.py:1181-1197`, `app/chat/service.py:216-217`
- **Суть:** история (10 сообщений) передаётся, но для расширения запроса
  берётся одно последнее user-сообщение. Детали обращения из первого
  сообщения (версия, код ошибки, конфигурация) выпадают из запроса уже на
  3–4 ходе.
- **Статус:** [ ]

### F15. Групповые чаты делят историю

- **Файлы:** `bot/handlers/_common.py:28-29`
- **Суть:** `owner_external_id = message.chat.id` — у группы одна общая
  история и контекст; follow-up одного человека склеивается с вопросом
  другого.
- **Статус:** [ ]

### F16. Ложные срабатывания ENCODING_PATTERNS; непонятные ошибки

- **Файлы:** `app/services/security/input_validator.py:26-30`
- **Суть:** `[A-Za-z0-9+/]{48,}` и base64-паттерны могут заблокировать
  вставленные license-ключи, хэши, подписи ошибок. Причина блокировки
  пользователю не раскрывается.
- **Статус:** [ ]

### F17. Lexical recall только русский; токены <3 символов отбрасываются

- **Файлы:** `app/services/rag.py:640-671`, `960-969`
- **Суть:** snowball-стеммер русский — англоязычные запросы по английским
  мануалам почти без lexical-помощи. Токены <3 символов («3D», «2D»)
  отбрасываются из lexical-терминов.
- **Статус:** [ ]

### F18. RAG-ответ не стримится; общий request_timeout=30с

- **Файлы:** `app/chat/service.py:230-234`, `app/services/rag.py:723-730`,
  `app/core/config.py:48-51`
- **Суть:** `_answer_with_rag` ждёт полный ответ → «Готовлю ответ…» висит до
  конца. `LLM_REQUEST_TIMEOUT` применён и к retrieval-embedding, и к
  генерации — сложный запрос с fallback-вариантами может упереться в 30 с.
- **Статус:** [ ]

### F19. `chat.system_prompt` игнорируется в RAG-пути

- **Файлы:** `app/chat/service.py:219-243`, `app/services/rag.py:39-48`
- **Суть:** для RAG всегда хардкод `RAG_SYSTEM_PROMPT`; кастомный промпт
  чата не применяется ни в RAG, ни (по сути) в non-RAG fallback, где у
  бот-чатов он `None`.
- **Статус:** [ ]

### F20. Два параллельных векторных хранилища

- **Файлы:** `app/main.py:115-121`, `app/core/config.py:144-153`, `224-233`
- **Суть:** чат-RAG использует `RAG_COLLECTION=rag_eval_chunk_1024`, агентные
  роутеры — `QDRANT_COLLECTION=documents` через отдельный `VectorStore`.
  Разные коллекции и разные сервисы — риск рассинхронизации и путаницы.
- **Статус:** [ ]

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
