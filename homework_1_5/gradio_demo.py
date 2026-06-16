from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


MODEL_ID = "tabularisai/multilingual-sentiment-analysis"


def normalize_label(label: str) -> str:
    value = label.lower()
    if "1 star" in value or "2 stars" in value:
        return "negative"
    if "3 stars" in value:
        return "neutral"
    if "4 stars" in value or "5 stars" in value:
        return "positive"
    if "negative" in value:
        return "negative"
    if "neutral" in value:
        return "neutral"
    if "positive" in value:
        return "positive"
    return value


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    local_files_only=True,
)
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
)


def classify(text: str) -> dict[str, object]:
    result = classifier(text)[0]
    return {
        "sentiment": normalize_label(result["label"]),
        "raw_label": result["label"],
        "score": round(float(result["score"]), 4),
    }


if __name__ == "__main__":
    import gradio as gr

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Textbox(lines=4, label="Русский отзыв или обращение"),
        outputs=gr.JSON(label="Sentiment"),
        title="Support Copilot Sentiment Demo",
        description="Локальное Gradio-демо лучшей HuggingFace-модели для анализа тональности support-отзывов.",
        examples=[
            ["Инженер быстро помог разобраться с лицензией, спасибо!"],
            ["Третий день жду ответ, работа стоит."],
            ["Отправляю лог-файл и скриншот ошибки для диагностики."],
        ],
    )
    demo.launch(server_name="127.0.0.1", server_port=7860)
