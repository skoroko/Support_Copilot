"""Export only a named evaluation's exact persisted graph evidence, without LLM calls."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "app" / "__init__.py").is_file() and (Path.cwd() / "app" / "__init__.py").is_file():
    ROOT = Path.cwd()  # Also supports a temporary script copied into the app container.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.agent_persistent import (
    _checkpointer_backend,
    _postgres_uri,
    _sqlite_path,
)


@asynccontextmanager
async def saver_for_deployment():
    backend = _checkpointer_backend()
    if backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(_postgres_uri()) as saver:
            yield saver
    elif backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(str(_sqlite_path())) as saver:
            yield saver
    else:
        raise ValueError("A separate process cannot read in-memory checkpoints")


def evaluation_threads(run: dict) -> dict[str, dict]:
    run_id = run["run_id"]
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Invalid evaluation run ID")
    threads = {}
    for row in run["results"]:
        owner, chat_id = row["owner"], row.get("chat_id")
        if not owner.startswith(f"eval-{run_id}-"):
            raise ValueError("Refusing to export a non-evaluation owner")
        if not chat_id:
            continue
        material = f"telegram\0{owner}\0{chat_id}".encode()
        thread_id = "chat-agent-" + hashlib.sha256(material).hexdigest()
        thread = threads.setdefault(thread_id, {"owner": owner, "chat_id": chat_id, "scenario_ids": []})
        thread["scenario_ids"].append(row["id"])
    return threads


def message_record(message) -> dict:
    record = {"type": message.type, "id": message.id, "content": message.content}
    if message.type == "ai":
        record["tool_calls"] = message.tool_calls
        record["finish_reason"] = message.response_metadata.get("finish_reason")
        record["usage"] = message.usage_metadata
    elif message.type == "tool":
        record.update(name=message.name, tool_call_id=message.tool_call_id)
    return record


def audit_pdf_pages(run_path: Path, data_dir: Path, output: Path) -> dict:
    """Check cited physical pages against local PDFs, independently of semantic scoring."""
    from pypdf import PdfReader

    run = json.loads(run_path.read_bytes())
    files = {}
    for path in data_dir.rglob("*.pdf"):
        files.setdefault(path.name, []).append(path)
    readers, rows = {}, {}
    citation_count = 0
    for row in run["results"]:
        answers = [("brief", row.get("sources", [])),
                   ("expanded", row.get("expansion", {}).get("result", {}).get("sources", []))]
        for mode, sources in answers:
            for source in sources:
                name = source.get("file_name", "")
                if not name.lower().endswith(".pdf"):
                    continue
                citation_count += 1
                key = source.get("chunk_id") or source.get("id") or f"{name}:{source.get('page')}"
                if key in rows:
                    rows[key]["used_by"].append(f"{row['id']}:{mode}")
                    continue
                audit = {"chunk_id": key, "file_name": name, "page": source.get("page"),
                         "printed_page": source.get("printed_page"), "used_by": [f"{row['id']}:{mode}"]}
                rows[key] = audit
                matches = files.get(name, [])
                if len(matches) != 1:
                    audit["status"] = "missing_or_ambiguous_file"
                    continue
                path = matches[0]
                if name not in readers:
                    readers[name] = PdfReader(path)
                reader = readers[name]
                page = source.get("page")
                if not isinstance(page, int) or not 1 <= page <= len(reader.pages):
                    audit["status"] = "invalid_physical_page"
                    continue
                text = reader.pages[page - 1].extract_text() or ""
                normalize = lambda value: "".join(char.casefold() for char in value if char.isalnum())
                excerpt = normalize(str(source.get("snippet") or ""))[:160]
                audit["excerpt_prefix_matches_page"] = bool(excerpt) and excerpt in normalize(text)
                printed = source.get("printed_page")
                audit["printed_page_in_header"] = bool(re.search(rf"(?<!\d){printed}(?!\d)", text[:500])) if printed is not None else None
                audit["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                audit["status"] = "verified" if audit["excerpt_prefix_matches_page"] and audit["printed_page_in_header"] is not False else "needs_review"
    result = {"run_id": run["run_id"], "pdf_citation_count": citation_count,
              "unique_pdf_chunks": len(rows), "checks": list(rows.values()),
              "note": "Physical page, printed header and excerpt prefix checks; not semantic faithfulness."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


async def export(run_path: Path, output: Path) -> dict:
    raw = run_path.read_bytes()
    run = json.loads(raw)
    result = {"schema_version": 1, "run_id": run["run_id"],
              "exported_at": datetime.now(UTC).isoformat(),
              "run_sha256": hashlib.sha256(raw).hexdigest(),
              "note": "Exact saved graph messages and tool contexts; no retrieval or model call repeated.",
              "threads": {}}
    async with saver_for_deployment() as saver:
        for thread_id, identity in evaluation_threads(run).items():
            checkpoint = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
            if checkpoint is None:
                result["threads"][thread_id] = {**identity, "status": "missing"}
                continue
            values = checkpoint.checkpoint["channel_values"]
            result["threads"][thread_id] = {
                **identity, "status": "exported", "checkpoint_id": checkpoint.checkpoint["id"],
                "checkpoint_ts": checkpoint.checkpoint["ts"],
                "messages": [message_record(message) for message in values.get("messages", [])],
                "knowledge_status": values.get("knowledge_status"),
                "answer_mode": values.get("answer_mode"),
                "answer_evidence": values.get("answer_evidence", []),
                "citation_evidence": values.get("citation_evidence", []),
                "sources": values.get("sources", []),
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-data-dir", type=Path, help="Run offline PDF page audit instead of checkpoint export")
    args = parser.parse_args()
    try:
        if args.pdf_data_dir:
            result = audit_pdf_pages(args.run, args.pdf_data_dir, args.output)
            print(json.dumps({"pdf_citation_count": result["pdf_citation_count"],
                              "unique_pdf_chunks": result["unique_pdf_chunks"],
                              "needs_review": sum(row["status"] != "verified" for row in result["checks"])}))
            return
        result = asyncio.run(export(args.run, args.output))
    except Exception as exc:  # noqa: BLE001 - never print a connection exception containing credentials
        # Connection exceptions can contain credentials. Print only their type.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps({"run_id": result["run_id"], "threads": len(result["threads"]),
                      "missing": sum(row["status"] == "missing" for row in result["threads"].values())}))


if __name__ == "__main__":
    main()
