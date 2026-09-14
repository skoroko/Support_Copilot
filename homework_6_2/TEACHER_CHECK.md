# Проверка домашнего задания 6.2

1. Выполнить офлайн-проверку:

   ```powershell
   $env:PHOENIX_WORKING_DIR="$PWD\var\phoenix-test"
   .venv-eval\Scripts\python.exe -m pytest -q tests/test_agent_react.py
   ```

2. В `app/services/agent_react.py` проверить нативный вызов Chat Completions:
   `tools=TOOLS`, `tool_choice="auto"`, `parallel_tool_calls=False`. Убедиться,
   что несколько tool calls безопасно останавливают выполнение.
3. В `app/services/agent_react_tools.py` проверить, что у каждого name-as-verb
   tool есть описание применения/входа/выхода/ограничений, `strict: true` и
   `additionalProperties: false`, а handlers импортированы из baseline 6.1.
4. Проверить единый deadline main LLM → tool → critic и точные ответы
   `Timeout` / `Превышен лимит итераций`; диапазоны по умолчанию и валидация —
   10 (8–20) итераций и 10 секунд (5–15).
5. Проверить trace: каждая итерация логирует номер, tool/args/observation,
   latency и tokens; `usage` суммирует main и critic вызовы. Отдельный critic
   допускает только `OK` или `REVISE: причина`, применяется не больше двух
   раз и добавляет feedback в следующий main шаг.
6. Проверить optional premium-path: `--premium-after-revision` включает
   `gpt-5.4` только для main шага после revision, основной default остаётся
   `gpt-5.4-mini` / `OPENAI_MODEL`.
7. Live-прогон после явного разрешения внешнего API выполнен командой:

   ```powershell
   python scripts/run_agent_comparison.py --allow-external-api
   ```

   Пять JSON из `docs/agent-comparison-results/` сверены с таблицей в
   `docs/agent-react-report.md`: naive — 11 шагов / 5 823 tokens, ReAct —
   12 итераций / 10 989 tokens, применено 2 revisions.
8. Проверить секреты только безопасной командой
   `python scripts/check_env_safety.py`; она выводит лишь статусы
   `configured`/`missing`. Её защита от возврата значений покрыта
   `tests/test_env_safety.py`.
9. Согласованность пяти JSON с Markdown-таблицей и итогами автоматически
   проверяется в `tests/test_agent_comparison_report.py`.
