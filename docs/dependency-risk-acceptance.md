# Временные исключения по зависимостям

Дата решения: 2026-09-22. Пересмотр не позднее: 2026-10-22.

## NLTK / PYSEC-2026-3740 / GHSA-8mgp-746c-j5xp / CVE-2026-81726

`llama-index-core` транзитивно требует NLTK, а исправленной версии на дату
решения нет. В production-коде Support Copilot не используются затронутые
`TransitionParser`, `AveragedPerceptron`, `PerceptronTagger` и
`save_maxent_params`; недоверенные pickle/model-файлы NLTK не загружаются.
Исключение действует только до появления исправленной версии или даты
пересмотра — что наступит раньше. При каждом обновлении `uv.lock` выполняется
повторный audit.

## RAGAS / diskcache

RAGAS и его evaluation-зависимости находятся только в optional extra `eval` и
не устанавливаются в release image. Evaluation запускается на копии
обезличенного корпуса без production-секретов, публичного listener и
недоверенных cache/artifact-файлов. Исключение пересматривается при каждом
обновлении `uv.lock` и прекращается при публикации исправления.
Смежные исправимые находки закрыты: `langchain-openai` обновлён до ветки
`>=1.1.14`, а старый Python-стек Phoenix, который привносил уязвимый
`strawberry-graphql`, удалён из графа; Phoenix остался изолированным
контейнером, а в приложение входит только OTLP-клиент.

## Закрытые решения

PyMuPDF удалён из runtime graph. PDF ingestion использует pypdf-backed
`llama_index.readers.file.PDFReader`, поэтому отдельное решение по
AGPL/commercial-лицензии PyMuPDF для этого проекта больше не требуется.
