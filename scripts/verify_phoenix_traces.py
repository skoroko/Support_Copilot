from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local Phoenix RAG traces")
    parser.add_argument("--url", default="http://127.0.0.1:6006")
    parser.add_argument("--project", default="support-copilot-rag")
    parser.add_argument("--minimum", type=int, default=20)
    args = parser.parse_args()

    response = httpx.get(
        f"{args.url.rstrip('/')}/v1/projects/{args.project}/traces",
        params={"limit": 1000, "include_spans": "true"},
        timeout=30.0,
    )
    response.raise_for_status()
    traces = list(response.json().get("data", []))
    request_traces = [
        trace
        for trace in traces
        if any(span.get("name") == "RAGService.answer" for span in trace.get("spans", []))
    ]
    complete_request_traces = []
    for trace in request_traces:
        spans = list(trace.get("spans", []))
        names = {str(span.get("name") or "") for span in spans}
        has_retriever = any("Retriever" in name for name in names)
        has_embedding = any(
            str(span.get("span_kind") or "").upper() == "EMBEDDING"
            for span in spans
        )
        has_llm = any(
            str(span.get("span_kind") or "").upper() == "LLM" for span in spans
        )
        if has_retriever and has_embedding and has_llm:
            complete_request_traces.append(trace)
    result = {
        "project": args.project,
        "request_trace_count": len(request_traces),
        "complete_request_trace_count": len(complete_request_traces),
        "all_trace_count": len(traces),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if len(request_traces) < args.minimum:
        raise SystemExit(
            f"Phoenix contains {len(request_traces)} RAG request traces; expected {args.minimum}+"
        )
    if len(complete_request_traces) < args.minimum:
        raise SystemExit(
            "Phoenix contains "
            f"{len(complete_request_traces)} complete RAG trace trees with "
            f"retriever, embedding and LLM spans; expected {args.minimum}+"
        )


if __name__ == "__main__":
    main()
