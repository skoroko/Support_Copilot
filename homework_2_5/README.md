# ДЗ 2.5: CLI-помощник техподдержки

Основной запуск:

```powershell
python -m support_copilot
```

Покрыто:

- system prompt по РРФО из ДЗ 2.2;
- few-shot примеры;
- retry/fallback через `RobustLLMClient`;
- cache через `LLMCache`;
- команды `/help`, `/clear`, `/stats`, `/quit`;
- история последних 10 сообщений;
- бонус: классификация FAQ / technical issue / complaint, prompt injection;
- бонус: эскалация;
- бонус: логирование в файл;
- мультимодальная команда `/image path/to/file.png [question]`.

Пример внутри запущенного CLI:

```text
/image homework_2_6\examples\conv.png что видно?
```

Команду запуска `python -m support_copilot` нужно вводить в PowerShell `PS ...>`,
а не внутри приглашения CLI `you>`.

Демо-лог 5+ сообщений:

```powershell
python homework_2_5/support_cli_demo.py
```
