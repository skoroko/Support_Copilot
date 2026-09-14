from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402


DEFAULT_OUTPUT = ROOT / "tests" / "eval" / "golden_dataset_raw.csv"
DEFAULT_FILES = (
    "02_Пособие по расчетам_midas GTS NX_Глава 2. Узел_Степень свободы_Система координат.pdf",
    "03_Пособие по расчетам_midas GTS NX_Глава 3. Элементы.pdf",
    "04_Пособие по расчетам_midas GTS NX_Глава 4. Материалы.pdf",
    "05_Пособие по расчетам_midas GTS NX_Глава 5. Алгоритмы.pdf",
)


def _sample_evenly(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    indexes = {round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)}
    return [items[index] for index in sorted(indexes)]


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    from ragas.testset.transforms.default import num_tokens_from_string

    if num_tokens_from_string(text) <= max_tokens:
        return text
    low, high = 1, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if num_tokens_from_string(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def select_documents(
    data_dir: Path,
    file_names: tuple[str, ...],
    pages_per_file: int,
    max_tokens_per_page: int,
):
    from llama_index.core import Document, SimpleDirectoryReader

    selected = []
    input_files = []
    for file_name in file_names:
        matches = [path for path in data_dir.rglob(file_name) if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one match for {file_name!r}, found {len(matches)}"
            )
        input_files.append(str(matches[0]))
    loaded = SimpleDirectoryReader(
        input_files=input_files,
        required_exts=[".pdf"],
        raise_on_error=True,
    ).load_data()
    by_file: dict[str, list] = {}
    for document in loaded:
        file_name = str((document.metadata or {}).get("file_name") or "")
        by_file.setdefault(file_name, []).append(document)
    missing = sorted(
        name for name in file_names if name.casefold() not in {key.casefold() for key in by_file}
    )
    if missing:
        raise FileNotFoundError(f"documents were not loaded: {missing}")
    for file_name in file_names:
        pages = sorted(
            by_file[file_name],
            key=lambda item: int((item.metadata or {}).get("page") or 0),
        )
        for page in _sample_evenly(pages, pages_per_file):
            # RAGAS 0.4.3's default long-document branch can run
            # HeadlineSplitter on nodes for which headline extraction failed.
            # Bounded page excerpts select its stable 101-500 token branch.
            selected.append(
                Document(
                    text=_truncate_to_tokens(str(page.text or ""), max_tokens_per_page),
                    metadata=dict(page.metadata or {}),
                )
            )
    return selected


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate raw RAGAS 0.4 testset candidates")
    parser.add_argument("--data-dir", type=Path, default=settings.rag_data_dir)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=34)
    parser.add_argument("--pages-per-file", type=int, default=3)
    parser.add_argument("--max-tokens-per-page", type=int, default=450)
    parser.add_argument("--model", default=os.getenv("RAG_EVAL_JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAG_EVAL_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--file", action="append", dest="files")
    parser.add_argument("--allow-external-data-transfer", action="store_true")
    args = parser.parse_args()
    if not args.allow_external_data_transfer:
        parser.error("--allow-external-data-transfer is required")
    if args.size < 30:
        parser.error("--size must be at least 30")
    if args.pages_per_file < 1 or not 120 <= args.max_tokens_per_page <= 500 or args.workers < 1:
        parser.error("pages/workers must be positive; max tokens must be in [120, 500]")
    args.files = tuple(args.files or DEFAULT_FILES)
    return args


def main() -> None:
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator

    args = parse_args()
    settings = get_settings()
    api_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    documents = select_documents(
        args.data_dir,
        args.files,
        args.pages_per_file,
        args.max_tokens_per_page,
    )
    llm = OpenAI(model=args.model, temperature=0, api_key=api_key)
    embeddings = OpenAIEmbedding(model_name=args.embedding_model, api_key=api_key)
    generator = TestsetGenerator.from_llama_index(
        llm,
        embeddings,
        llm_context=(
            "Техническая документация MIDAS GTS NX. Формулируй вопросы и эталонные "
            "ответы по-русски, не добавляй факты вне предоставленных документов."
        ),
    )
    testset = generator.generate_with_llamaindex_docs(
        documents,
        testset_size=args.size,
        run_config=RunConfig(max_workers=args.workers, max_retries=3, timeout=180, seed=42),
        raise_exceptions=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    testset.to_pandas().to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Generated {len(testset.samples)} candidates from {len(documents)} document pages")
    print(args.output)


if __name__ == "__main__":
    main()
