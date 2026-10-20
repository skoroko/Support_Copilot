# PR52: публичные runtime-эксперименты, 20 октября 2026

Ни один полностью проверенный cohort ниже не выполнил исходный содержательный gate **8/8**. Быстрые варианты с пропусками не считаются успешным ускорением. Это восемь development-вопросов, **не52**, не sealed holdout и не доказательство P95/SLA. Производственный релиз этим отчётом не подтверждается.

Числовой источник: [runtime_cohorts.json](../tests/eval/results/2026-10-20_catalog_inventory_pilot/runtime_cohorts.json). Исторические [v1/v2](catalog_inventory_2026-10-20.md) и [summary.json](../tests/eval/results/2026-10-20_catalog_inventory_pilot/summary.json) сохранены без изменения. Полные публичные ответы остаются в ignored `var/catalog_inventory_eval`; здесь только метаданные и независимые выводы.

## Протокол и неизменный gate

Фиксированный порядок: RU, EN, VAR,01,07,10,16,32. Три inventory-вопроса требуют все26 Ground O, отдельно подтверждённые NorSand/SCLAY1S, буквальные O/X/поведение и неизвестную версию. Controls требуют шесть DOF; назначение/Kn/Kt/Coulomb; предпосылки/параметры/ограничения Мора—Кулона с разбором неоднозначности источника; Initial/NR/Modified/BFGS, оговорку реализации, LineSearch/Overrelaxation; Review/Draining/NodalHead Total-Pressure-time/Transmitting-frequency и механические закрепления. Полнота не ослабляется ради времени. Reference slots механически обоснованы, но ожидают инженерного принятия.

PASS IDs берутся из независимых reviewer turns; для v10/v11/v13 также сохранены отдельные JSON reviews с SHA в числовом отчёте. Unknown означает отсутствие полного вердикта, а не PASS. Прежние неудачи и неполные прогоны не удалены.

## Времена всех наблюдений

Секунды в одинаковом фиксированном порядке. Для partial/error строк пустые места не заменены другими вопросами. Median — только описательная статистика наблюдавшихся строк.

| Cohort | RU | EN | VAR | 01 | 07 | 10 | 16 | 32 | min / median / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_quiet | 24.314 | 26.698 | 55.775 | 36.985 | 6.355 | 8.567 | 20.945 | 28.728 | 6.355 / 25.506 / 55.775 |
| v1 | 22.201 | 22.483 | 21.691 | 16.074 | 11.424 | 7.154 | 11.709 | 21.102 | 7.154 / 18.588 / 22.483 |
| v2 | 55.381 | 7.159 | 19.127 | 18.586 | 9.166 | 39.539 | 47.548 | 24.286 | 7.159 / 21.706 / 55.381 |
| v3_off | 10.349 | 6.103 | 7.906 | 8.639 | 6.788 | 14.727 | 9.487 | 10.400 | 6.103 / 9.063 / 14.727 |
| v4_decisions_audit | 22.316 | 13.251 | 22.567 | 23.623 | 12.434 | 13.422 | 15.543 | 20.058 | 12.434 / 17.800 / 23.623 |
| v4_responses_audit | 27.602 | 10.731 | 33.418 | 22.230 | 9.264 | 24.444 | 33.986 | 50.590 | 9.264 / 26.023 / 50.590 |
| v5_luna_off | 10.000 | 8.814 | 17.346 | 4.125 | 21.262 | 15.222 | 19.836 | 16.773 | 4.125 / 15.997 / 21.262 |
| v5_sol_off | 14.788 | 7.550 | 15.539 | 4.833 | 13.374 | 19.285 | 11.436 | 25.564 | 4.833 / 14.081 / 25.564 |
| v6_sol_off | 49.551 | 12.906 | 19.301 | 10.150 | 9.792 | 22.147 | 14.859 | 28.531 | 9.792 / 17.080 / 49.551 |
| v7_decisions_partial | 50.228 | 12.222 | — | — | — | — | — | — | 12.222 / 31.225 / 50.228 |
| v8_luna_decisions_selection | 38.826 | 56.169 | 39.284 | 9.167 | 16.263 | 20.101 | 29.475 | 59.443 | 9.167 / 34.151 / 59.443 |
| v9_sol_decisions_selection | 17.373 | 21.610 | 46.712 | 18.746 | 16.050 | 21.953 | 24.126 | 31.913 | 16.050 / 21.782 / 46.712 |
| v10_luna_off | 30.865 | 28.408 | 19.199 | 6.677 | 17.471 | 13.058 | 19.846 | 16.574 | 6.677 / 18.335 / 30.865 |
| v11_sol_off | 20.181 | 14.124 | 34.137 | 12.072 | 17.897 | 26.375 | 16.753 | 38.562 | 12.072 / 19.039 / 38.562 |
| v11_luna_off | 39.102 | 12.874 | 20.123 | 7.005 | 18.051 | 14.300 | 29.180 | 43.338 | 7.005 / 19.087 / 43.338 |
| v11_luna_decisions_failed | 16.926 | — | — | — | — | — | — | — | 16.926 / 16.926 / 16.926 |
| v13_luna_off | 15.917 | 19.643 | 27.424 | 7.089 | 16.193 | 12.514 | 19.119 | 31.209 | 7.089 / 17.656 / 31.209 |
| v14_luna_off | 24.223 | 18.868 | 23.174 | 9.639 | 17.431 | 17.434 | 24.088 | 38.574 | 9.639 / 21.021 / 38.574 |
| v14_luna_decisions_selection | 53.334 | 42.250 | 34.919 | 8.406 | 23.532 | 15.671 | 26.765 | 27.882 | 8.406 / 27.3235 / 53.334 |

## Модель, режим и качество

Модель/reasoning/квоты — запись конфигурации выполнения от root, **не утверждение о self-reported model ID**: raw model metrics экспортируют tier, но не model_id/reasoning. Во всех успешных model calls указан tier `fast`; стоимость здесь не рассчитывается. Для незаписанной конфигурации v7 оставлено unknown. OFF в именах поздних cohorts относится к API selector/auditor: catalog/physical RAG дополнения в этих кандидатах включены.

| Cohort | Answer model / reasoning | Primary + follow-up chars | Answer audit | Table selector | Human content PASS IDs |
|---|---|---|---|---|---|
| baseline_quiet | gpt-6-luna / medium | 12000 + 6000 | off | off | unknown |
| v1 | gpt-6-luna / medium | 12000 + 6000 | off | off | unknown |
| v2 | gpt-6-luna / medium | 12000 + 6000 | off | off | unknown |
| v3_off | gpt-6-luna / medium | 12000 + 6000 | off | off | 2/8: RU, 10 |
| v4_decisions_audit | gpt-6-luna / medium | 12000 + 6000 | decisions | off | 5/8: RU, EN, VAR, 07, 10 |
| v4_responses_audit | gpt-6-luna / medium | 12000 + 6000 | responses | off | 3/8: RU, EN, 16 |
| v5_luna_off | gpt-6-luna / medium | 12000 + 6000 | off | off | 3/8: EN, 01, 10 |
| v5_sol_off | gpt-6.1-sol / medium | 12000 + 6000 | off | off | 6/8: RU, EN, VAR, 01, 07, 10 |
| v6_sol_off | gpt-6.1-sol / medium | 12000 + 6000 | off | off | 6/8: EN, VAR, 01, 07, 10, 32 |
| v7_decisions_partial | unknown / unknown | unknown | off | decisions | unknown |
| v8_luna_decisions_selection | gpt-6-luna / high | 12000 + 6000 | off | decisions | 5/8: RU, VAR, 01, 07, 10 |
| v9_sol_decisions_selection | gpt-6.1-sol / medium | 12000 + 6000 | off | decisions | 7/8: RU, EN, VAR, 01, 07, 10, 16 |
| v10_luna_off | gpt-6-luna / medium | 12000 + 6000 | off | off | 6/8: RU, EN, 01, 07, 10, 16 |
| v11_sol_off | gpt-6.1-sol / medium | 12000 + 6000 | off | off | 6/8: RU, EN, VAR, 01, 07, 10 |
| v11_luna_off | gpt-6-luna / medium | 12000 + 6000 | off | off | 6/8: RU, VAR, 01, 07, 10, 16 |
| v11_luna_decisions_failed | gpt-6-luna / medium | 12000 + 6000 | off | decisions | unknown |
| v13_luna_off | gpt-6-luna / medium | 24000 + 18000 | off | off | 5/8: RU, EN, 01, 07, 16 |
| v14_luna_off | gpt-6-luna / medium | 24000 + 18000 | off | off | 7/8: RU, VAR, 01, 07, 10, 16, 32 |
| v14_luna_decisions_selection | gpt-6-luna / medium | 24000 + 18000 | off | decisions | 4/8: EN, VAR, 01, 07 |

v1: минимум известны FAIL EN(version2021 из filename),01(DOF5/6),16(оговорка/Overrelaxation). v2: известны FAIL RU(SCLAY1S) и01(DOF5/6). Полного PASS-ID вердикта8/8 исторический summary не содержит; остальные строки не объявлены PASS. Baseline — reference для пяти controls, не подтверждённый all26 inventory-победитель.

| Cohort | Конкретные потери / ограничения |
|---|---|
| v3_off | EN: утверждение версии; variant: неверное поведение Hyperbolic; 01: DOF5/6; 07: Kn/Kt; 16: алгоритмы; 32: Draining. |
| v4_decisions_audit | 01: DOF5/6; 16: BFGS/Overrelaxation/оговорка реализации; 32: Review/Head. |
| v4_responses_audit | Variant: Hyperbolic/пропуск; 01: max-iteration handoff; 07: Kn/Kt; 10: source/переводная неоднозначность; 32: Review/Draining. |
| v5_luna_off | RU: дополнительные модели; variant: четыре anisotropic/equivalent; 07: отрицание Kt; 16: методы; 32: Review/Head. |
| v5_sol_off | 16: Line Search/Overrelaxation; 32: Review/частотная зависимость. |
| v6_sol_off | RU: полная таблица отсутствует в packet; 16: Overrelaxation. |
| v7_decisions_partial | Только 2/8; полного независимого fixed-eight verdict нет. |
| v8_luna_decisions_selection | EN: четыре anisotropic/equivalent; 16: BFGS/принципы обновления; 32: Review. |
| v9_sol_decisions_selection | 32: Review. |
| v10_luna_off | Variant: Hyperbolic назван нелинейно-упругим вопреки буквальной таблице; 32: Review/Head. |
| v11_sol_off | 16: Overrelaxation/оговорка реализации; 32: Review. |
| v11_luna_off | EN: четыре anisotropic/equivalent; 32: Review и Constraint Basic/Advanced/Auto/all6. |
| v11_luna_decisions_failed | Одна error row OpenAIConnectionError, затем остановлен; это не полный source-selector A/B и не ответный quality verdict. |
| v13_luna_off | Variant: число25 при26 имёнах; 10: source/переводная неоднозначность; 32: Review/частотная зависимость/значения Total-Pressure. Отдельно EN нарушает русский контракт. |
| v14_luna_off | EN: обязательная оговорка unknown version полностью опущена; ложной версии нет. Остальные7 PASS; русский контракт8/8. |
| v14_luna_decisions_selection | RU: NorSand;10: неоднозначность главного напряжения;16: caveat ошибочно расширен на NR;32: Constraint/all6 и смыслы Total/Pressure. |

v11 Decisions: **OpenAIConnectionError** в единственной строке RU16.926s; запуск не завершил cohort и не является валидным source-selector A/B. v7 имеет только2/8 ответов. v12 — preparation-only probe16/32 при24K, без генерации ответов и fixed-eight quality verdict. Эти попытки сохранены отдельно от полных cohorts.

В v13 исходный content gate **5/8**. Английский EN ответ отдельно нарушает существующий продуктовый контракт русского вывода: combined **4/8**. Язык раньше не был явным пунктом fixed-eight score; теперь он одинаково проверен для всех восьми, а прежние оценки задним числом не переписаны. Вариантный ответ v13 перечисляет26 имён, но заявляет25 — ошибка счёта, не потеря строки. В32 Review preview виден, но не выбран для чтения; полный Transmitting с частотной пружиной/демпфером получен и процитирован, но аспект опущен. Outbound model packets не записаны: наличие в final-call packet реконструировано по frozen-коду и state, а не raw capture.

## Decisions: два разных применения

**Answer audit v4** проверяет уже написанный черновик четырьмя общими predicates: requested_parts, enumeration, grounding, scope. Это не выбор таблицы. В freely generated graph cohort Decisions было16completed audits, по2 на case:8 повторно проверенных revisions; финальные audit-reject у всех5 human-PASS ответов. Responses:13completed audits,5 повторно проверенных revisions; 01 дошёл до88-символьного max-iteration handoff при6modelcalls/5retrievals;32 —50.590s,6modelcalls/13reportedretrievals. Ошибки верхнего уровня отсутствуют, но это не отменяет semanticFAIL. Число revisions не равно числу поисков/modelcalls.

В дополнительном same-packet answer-audit A/B на8 зафиксированных финальных answer/evidence SHA совпадают между API. Decisions accepted0/8, отклонил все5 human-good; Responses accepted1/8, отклонил4из5 human-good. False accepts среди3 human-bad не наблюдались. Median audit duration0.4605s против4.5735s. Это показывает расхождение с human gate и непроверенный threshold; probability/confidence не равна качеству. Из этих8 нельзя получить общую accuracy или причинный выигрыш готовых ответов.

**Source selection** — отдельный контролируемый опыт реального TableSelectionService. Общие question/candidate descriptions byte-identical, inputSHA сверены;97 validated catalog groups в exact GTSNX/unknown partition десяти источников, после <=6K filter96 candidates. Восемь prompts повторены дважды с counterbalanced order/fresh clients; два EN rows на endpoint обработаны lexically без provider call. Responses сохраняет реальный timeout5s; ошибки не подавлялись.

| Source choice | Correct /16 incl skips | Real provider correct / calls | Median service s | Range service s |
|---|---:|---:|---:|
| lexical | 4/16 | 0/0 | — | — |
| off | 4/16 | 0/0 | — | — |
| decisions | 14/16 | 12/14 | 0.9130980 | 0.653583–1.616627 |
| responses | 16/16 | 14/14 | 2.6203795 | 2.140298–3.616002 |

Decisions **14/16**, то есть12/14 реальных вызовов, median **0.913098s**; Responses **16/16**, то есть14/14, median **2.6203795s**. Decisions быстрее в14/14 real pairs, но дважды неверно выбрал Property с типами элементов вместо constitutive-material inventory для paraphrase-ru-2. Source/hash revalidation исключает выдуманную таблицу, но не доказывает релевантность выбранного источника и полноту ответа. Local lexical/off4/16 используют исходный запрос без производственного bilingual translation; это не production RAG recall. В v9 исходный RU имел ноль anchors в isolated fixture, но actual initial search имел перевод и1anchor: нельзя приписывать его успешную таблицу Decisions.

Один целевой каталог и два повтора не дают общей оценки routing. Ни answer quality, ни end-to-end speedup/cost/P95 не измерены этим source-choice опытом. Prospective source-choice **v2 ещё не выполнен**; gates должны быть объявлены до новых вызовов. Автоактивация на основании одной скорости не обоснована.

## Ограничения сравнения и сохранённые неудачи

- Свободные последовательные cohorts меняют LLM-переводы, retrieval packets, ответы, provider latency и код. Это не рандомизированное причинное сравнение моделей или API. Часть v4 пересекалась с полными CPU-тестами; поздние quiet runs — другие наблюдения.
- Model wall не включает provider работу внутри RAG. Retrieval включает backend/network/translation и локальные стадии; residual не равен CPU. Reported retrievals не считают exact-navigation reads.
- Бюджет вырос с12K+6K до24K+18K в v13; это существенное изменение входа, а не доказательство ускорения при одинаковом пакете. Кэш/прогрев/снимки тоже различаются.
- Исходный baseline32=430.547s сохранён исторически.415s вне суммы model/retrieval согласуются с блокировкой verbose stdout стенда; это гипотеза harness, не установленная provider latency рабочего бота. Quiet повтор32=28.728s.
- Wrong import/source paths ранних relocated запусков, partial cohorts и connection failure не превращены в PASS и не заменены удобными строками. Разрешение citations и unit-tests не доказывают ответную полноту.
- Технически подозрительное направление обжатия в v13 case07 буквально повторяет cited English source. Это source ambiguity для инженерного принятия, не объявленная новая галлюцинация.
- Нет утверждения об успехе52кейсов, sealed holdout, SLA/P95, validated confidence threshold или production readiness. Старые FAIL остаются видимыми.

v14 добавлен только после завершения всех8 ответов и независимого review; прежние результаты сохранены.


## Дополнение v14 Luna OFF

Snapshot app manifest от root: `dca11db25b0d5c4e62ec8b0eb677d48083a2e16bd5b295efd215a8cc89baa05c` (не пересчитан этим reviewer). По записи выполнения во время cohort app-код не менялся. Модель Luna medium Fast,24K+18K, answer-audit и source-selector OFF.

Исходный content gate **7/8**, русский контракт **8/8**, combined **7/8**. Единственный FAIL — EN: все26 названий и NorSand/SCLAY1S есть, но нет обязательной оговорки неизвестной версии при source metadata version=unknown. Это потеря scope-уточнения; выдуманной конкретной версии или явного current-version exhaustiveness не найдено. Нельзя объявлять8/8, незаметно отменив unknown-version facet.

В32 проверены точные ссылки: Review[13]→CHM Review.htm (неизвестная кривая депрессии/итерации), Head[11]→Nodal_Head.htm (Total относительно начала координат, Pressure0 на поверхности воды, функция времени), Transmitting[5]→AnalysisMethod.pdf page64 (частотная пружина/демпфер), constraints[15]→page36 (Basic/Advanced/Auto/all6). Множитель функции не повторён; он не введён задним числом как новый обязательный facet. В10 discrepancy intermediate/среднее теперь явно отмечена с верными English/Russian sources. В16 есть caveat, BFGS, LineSearch и Overrelaxation. Variant верно сообщает26 и сохраняет буквальный Hyperbolic.

Min/median/max **9.639/21.021/38.574s**, менее10s1/8;20modelcalls,8reportedretrievals +12exact-navigation attempts. Citation IDs разрешаются и исключений нет, что не отменяет contentFAIL. Это отдельный sequential cohort, не причинное доказательство влияния aliases/column counts/prompt. Новые парные опыты могут только добавить доказательства; они не стирают этот gate failure.


## Дополнение v14 Decisions source selector

Тот же frozen app `dca11db25b0d5c4e62ec8b0eb677d48083a2e16bd5b295efd215a8cc89baa05c`, Luna medium Fast и24K+18K; изменён только конфигурационный table_selection=decisions, answer-audit OFF. **Content4/8, русский8/8, combined4/8**, gateFAIL. PASS EN,VAR,01,07. RU потерял NorSand (его нет и в saved evidence).10 без оговорки использует русское среднее главное при английском intermediate в packet.16 утверждает, что все три метода не представлены явно, хотя cited Algorithms.pdf25 исключает только constant/ModifiedNR; это неподдержанное расширение на NR.32 сохранил Review/Draining/frequency, но потерял Basic/Advanced/Auto/all6 и значения Total/Pressure.

Min/median/max **8.406/27.3235/53.334s**;23modelcalls,12reportedretrievals +15exact-navigation attempts. Исключений нет, citations resolve. Ни это, ни наличие таблицы не доказывает полноту.

В actual semantic_selection telemetry ровно **2** cache-miss API invocations без ошибок: RU retrieve1 **3626.581ms**,16 retrieve1 **1319.376ms**, вместе **4.945957s** service wall. В остальных экспортированных retrievals semantic-selection events нет. Поэтому нельзя целиком приписывать53.334/42.250/... этому флагу. Exact navigation — отдельные операции. Свободно сгенерированные queries/ответы и provider scheduling не спарены; это не доказательство причинного ухудшения или ускорения Decisions.

Root сообщил около15s pytest bootstrap в середине D-cohort: подготовка остановлена до продвижения tests; полный suite запущен только после завершения cohort. Краткое CPU/setup пересечение — дополнительный confound. Полный prospective paired-source v2 ещё не включён: его можно добавить после отдельного законченного опыта без переписывания сохранённых FAIL.
