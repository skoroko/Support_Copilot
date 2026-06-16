# Benchmark sentiment analysis

Задача: классификация тональности русских support-отзывов.

| Провайдер | Модель | Лицензия | Accuracy | Средняя latency |
| --- | --- | --- | ---: | ---: |
| HuggingFace Pipeline | `cointegrated/rubert-tiny-sentiment-balanced` | not specified | 0.600 | 0.0111 сек |
| HuggingFace Pipeline | `blanchefort/rubert-base-cased-sentiment` | not specified | 0.600 | 0.0329 сек |
| HuggingFace Pipeline | `tabularisai/multilingual-sentiment-analysis` | cc-by-nc-4.0 | 0.900 | 0.0208 сек |
| OpenRouter | `openai/gpt-oss-120b:free` | n/a | 1.000 | 3.2853 сек |

Лучший общий baseline по accuracy/latency: `openai/gpt-oss-120b:free`.
Лучшая HuggingFace-модель для Gradio-демо: `tabularisai/multilingual-sentiment-analysis`.
