# Техническая документация Support Copilot

Актуальный комплект для итоговой сдачи, 16 октября 2026 года.
Основной продукт — Telegram-ассистент с одним persistent LangGraph-агентом,
поиском по мануалам и подтверждением внешних действий.

## Порядок чтения

| Шаг | Материал | Что в нём искать |
|---:|---|---|
| 1 | [README готового продукта](../README.md) | Пользовательский сценарий, запуск, конфигурация и проверки |
| 2 | [Окончательное техническое описание](final_technical_documentation.md) | Компоненты системы и семь решений в формате «выбор → альтернативы → причина выбора» |
| 3 | [Финальный агентный граф](final_agent_graph.md) | Реальные узлы, переходы, interrupt/resume и путь «Подробнее» |
| 4 | [Объединённая система](unified_agent.md) | Подробные контракты интеграции, ограничения и датированные результаты проверок |

Схема доступна отдельно как [SVG](final-agent-graph.svg) и
[Mermaid, полученный из исполняемого графа](final-agent-graph.mmd).
Эти файлы описывают `app/services/agent_persistent.py`, используемый Telegram,
а не ранние учебные графы.

## Справочник по темам

| Тема | Документ | Область и статус |
|---|---|---|
| Telegram | [telegram_bot.md](telegram_bot.md) | Текущие меню, команды, feedback, approval и renderer |
| Retrieval и ingestion | [rag.md](rag.md) | Текущий поиск по мануалам, provenance, ограничения и эксплуатация |
| Общая обработка документации | [documentation_evidence.md](documentation_evidence.md) | Новый каталог PDF/CHM/HTML, общий retrieval-first путь, 120 новых eval-сценариев; release gate ещё не пройден |
| Корпус | [data_inventory.md](data_inventory.md) | Состав документов и отложенное пополнение |
| История диалогов | [chat.md](chat.md) | Repository, API и исходный sliding-window контракт; Telegram теперь использует graph adapter, см. `unified_agent.md` |
| Checkpoints и отправка | [agent-persistent-report.md](agent-persistent-report.md) | Устройство saver/HIL; Docker transcript относится к учебному снимку |
| Наблюдаемость | [observability/README.md](observability/README.md) | Логи, метрики и tracing |
| Качество финального агента | [telegram_agent_evaluation.md](telegram_agent_evaluation.md), [48 оценок текущего прогона](../tests/eval/results/2026-10-15_telegram_agent_self_weight_repeat/semantic_review.json) | Прогон 15.10.2026 на обновлённых Docker: собственный вес №24, №31, источники/страницы, уточнения, expanded, отказы, время и сбой №9 |
| Промежуточные попытки 15 октября | [Пропуск собственного веса](../tests/eval/results/2026-10-15_telegram_agent_semantic_fix/acceptance.json), [сбои API/лимит итераций](../tests/eval/results/2026-10-15_telegram_agent_self_weight_fix/acceptance.json) | Ответы, checkpoint и PDF-аудиты сохранены для сравнения; общие оценки качества этим попыткам не назначались |
| Предыдущая оценка после смысловых исправлений | [Исторический отчёт 14.10.2026](telegram_agent_evaluation_semantic_fix_2026-10-14.md) | №24 brief/expanded пропускали собственный вес; дополнительные ходы остановлены квотой API |
| Baseline перед смысловыми исправлениями | [telegram_agent_evaluation_citation_fix_2026-10-14.md](telegram_agent_evaluation_citation_fix_2026-10-14.md), [результаты](../tests/eval/results/2026-10-14_telegram_agent_citation_fix_verified/summary.json) | Предыдущая оценка с ошибками №24/31 сохранена без перезаписи |
| Baseline до исправления цитат | [telegram_agent_evaluation_baseline_2026-10-14.md](telegram_agent_evaluation_baseline_2026-10-14.md), [исходные результаты](../tests/eval/results/2026-10-14_telegram_agent/summary.json) | Первоначальный прогон сохранён без перезаписи JSON/CSV; не оценка исправленной версии |
| История качества RAG | [rag_evaluation.md](rag_evaluation.md), [golden dataset](../tests/eval/golden_dataset.json) | Эксперименты 12.09.2026, 34 вопроса; не оценка финального Telegram-агента |
| Выбор модели | [Сравнение двух облачных моделей](../homework_1_1/results/comparison_report.md), [локальная/облачная модель](../homework_1_4/results/benchmark_summary.json) | Ранние учебные замеры; границы приведены в итоговом описании |
| Runtime Luna и reasoning | [luna_runtime_evaluation.md](luna_runtime_evaluation.md) | 384 ответа 15.10.2026, mini против Luna none/low/medium/high/xhigh; выбран medium |
| Переключение бота на Luna | [luna_runtime_rollout.md](luna_runtime_rollout.md) | Локальный backend переключён 15.10.2026; 7 обычных ответов и 2 расширения, проверена история mini |
| Переключение RAG на Luna | [rag_luna_rollout.md](rag_luna_rollout.md) | Ответы medium, переформулирование none; проверка API, SSE и сохранения истории |
| Выбор embeddings | [ДЗ 5.1](../homework_5_1/README.md) | Обоснование large/1024; цены и внешние характеристики датированы |
| Выбор chunking | [chunking_experiment.md](chunking_experiment.md), [rag_evaluation.md](rag_evaluation.md) | Offline выбор 512/64, затем облачный эксперимент и переход на 1024/64 |
| Выбор vector DB | [vector_store.md](vector_store.md) | Qdrant и альтернативы; коллекция `documents` и HNSW-параметры учебного адаптера не являются настройками production RAG |
| Single-agent / multi-agent | [multi-agent-report.md](multi-agent-report.md) | Сравнение на пяти mock-KB вопросах и решение оставить single-agent |

## Исторические материалы

| Артефакт | Как использовать |
|---|---|
| [architecture.md](architecture.md) | Первый раздел актуален; четыре слоя и ADR ниже — исторический CLI-дизайн |
| [architecture-historical-cli.pdf](architecture-historical-cli.pdf) | **Исторический PDF архитектуры CLI.** Сохранён без переработки; не схема финальной системы |
| [agent-graph-report.md](agent-graph-report.md), [custom](agent-graph-custom.mmd), [prebuilt](agent-graph-prebuilt.mmd) | Учебные графы Б6.3, до объединения Telegram/RAG и persistent approval |
| [architecture-multi-agent.md](architecture-multi-agent.md) | Схема экспериментального supervisor, не выбранного для основного сценария |
| `homework_*`, ранние traces и benchmark-ответы | История накопительного проекта и доказательства конкретных экспериментов |
