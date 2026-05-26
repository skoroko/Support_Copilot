# ДЗ 2.2: промпт-инжиниринг

Скрипт: `prompt_engineering_demo.py`.

Покрыто:

- production-ready system prompt по РРФО;
- 4 few-shot примера;
- 5 автотестов `should_contain` / `should_not_contain`;
- тест prompt injection;
- подсчет `tiktoken`, целевой диапазон 300-800 токенов;
- сравнение с few-shot и без few-shot на 3 запросах.

Если локальный `tiktoken` пытается скачать BPE-файл, а сеть недоступна, скрипт
не падает и пишет в лог offline fallback для проверки диапазона. В окружении с
закешированной кодировкой будет использован настоящий `tiktoken`.

Запуск:

```powershell
python homework_2_2/prompt_engineering_demo.py --log-file homework_2_2/prompt_test_log.txt
```
