# Проверка домашнего задания 6.3

1. Убедиться, что `pyproject.toml` закрепляет диапазоны LangGraph/LangChain
   1.x, `uv.lock` фиксирует граф, а `.venv-eval` импортирует их без запуска модели.
2. В `app/services/agent_graph.py` проверить `AgentState`: reducer
   `add_messages`, scalar `iteration_count`, `operator.add` для JSON-подобных
   `tool_results`; клиент и ключ отсутствуют в состоянии.
3. Проверить три общих `@tool` с полезными docstring и тот же
   `SYSTEM_PROMPT`/model/tool-list у `custom_graph` и `prebuilt_graph`.
4. Проследить topology custom graph:
   `START → call_model → (execute_tool | force_finish)`,
   `execute_tool → call_model`, `force_finish → END`.
5. Запустить офлайн-тесты. Они замокают модель, проверят неизвестный tool,
   serializable tool state, выполнение custom/prebuilt graphs и остановку на
   шестом model turn без внешнего API.

   ```powershell
   .venv-eval\Scripts\python.exe -m pytest -q tests/test_agent_graph.py
   ```

6. Запустить `scripts/visualize_graph.py` и открыть оба `.mmd`: они должны
   соответствовать topology в коде.
7. Проверить, что `scripts/bench_agents.py` импортирует ровно `CASES` из
   `scripts/run_agent_comparison.py`, запрещает API без
   `--allow-external-api`, требует минимум три повтора и задаёт `thread_id`
   custom graph. После отдельного разрешения live прогон обязан дать 45 строк
   с измеренными latency/tokens/steps; до него report честно содержит pending.
8. Проверить, что baseline 6.2 (`agent_react.py`) не менялся в этом
   инкременте, и выполнять диагностику секретов только
   `scripts/check_env_safety.py`.
