from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def main() -> None:
    api = HfApi()
    models = list(
        api.list_models(
            filter="text-classification",
            search="sentiment russian",
            sort="downloads",
            direction=-1,
            limit=20,
            full=True,
        )
    )
    rows = []
    for model in models:
        tags = sorted(model.tags or [])
        rows.append(
            {
                "model_id": model.modelId,
                "downloads": getattr(model, "downloads", None),
                "likes": getattr(model, "likes", None),
                "pipeline_tag": getattr(model, "pipeline_tag", None),
                "license": (getattr(model, "cardData", None) or {}).get("license")
                if isinstance(getattr(model, "cardData", None), dict)
                else None,
                "tags": tags,
            }
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "hf_model_search.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} model search results to {RESULTS_DIR / 'hf_model_search.json'}")


if __name__ == "__main__":
    main()
