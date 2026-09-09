# Проверка ДЗ 5.4

1. `tests/eval/retrieval_dataset.json` содержит 28 вопросов и 1-3 эталонных документа на каждый вопрос.
2. Набор покрывает точечные факты, сравнения и многошаговые вопросы по 10 документам корпуса.
3. `app/services/chunking.py` реализует `fixed`, `recursive` и `semantic` стратегии.
4. Базовые коллекции имеют требуемые имена: `docs_fixed`, `docs_recursive`, `docs_semantic`.
5. `app/services/retrieval_eval.py` считает Hit Rate@5, MRR@10 и Recall@10 единой функцией.
6. `app/services/reranker.py` подключает локальный `BAAI/bge-reranker-v2-m3` через `sentence-transformers`.
7. `scripts/evaluate_chunking.py` переиндексирует коллекции, измеряет метрики и latency, выполняет четыре parameter experiments.
8. Полные числа сохранены в `homework_5_4/results/chunking_experiment.json` и сведены в `docs/chunking_experiment.md`.
9. BGE повысил MRR@10 с 0.771 до 0.839; Hit Rate@5 остался 0.929, Recall@10 снизился с 0.750 до 0.720.
10. Итоговая конфигурация `recursive/512/64/top-K=10` зафиксирована в `app/core/config.py` и `.env.example`.
11. Целевой Hit Rate@5 >= 0.5 выполнен: получено 0.929 на 28 вопросах.
12. Офлайн-режим не отправляет документы внешнему провайдеру; внешний embedding требует явного флага разрешения.
