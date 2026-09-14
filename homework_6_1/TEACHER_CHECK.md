# Проверка домашнего задания 6.1

1. Запустить `python -m pytest -q tests/test_agent_naive.py`.
2. Проверить `TOOLS` и явный allowlist `DISPATCH` в
   `app/services/agent_naive_tools.py`; динамического вызова функций нет.
3. Выполнить `python -m app.services.agent_naive "Сколько времени в Москве?" --trace`.
4. Убедиться, что неизвестный `get_user_balance` превращается в tool-result и
   цикл продолжается, а `max_steps` останавливает повторные вызовы.
5. Проверить, что `send_telegram_message` только печатает строку и не содержит
   Telegram-клиента или сетевого вызова.
6. После явного разрешения внешнего API выполнить
   `python scripts/run_naive_agent_cases.py --allow-external-api` и открыть пять
   JSON-файлов в `docs/agent-naive-traces/`.
