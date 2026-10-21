"""Verify diploma links, frozen evidence, runtime hashes and diagram consistency.

Offline only. Refreshes two documentation audit JSON files; no external calls.
Run from the repository: .venv/Scripts/python.exe scripts/verify_diploma_documentation.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    "README.md",
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").rglob("*.md")),
]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


portable = {
    p
    for p in git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    .decode()
    .split("\0")
    if p
}
issues: list[dict[str, str | int]] = []
link_count = 0
for name in DOCS:
    source = ROOT / name
    content = source.read_text(encoding="utf-8")
    content = re.sub(r"(?ms)^```.*?^```[^\n]*", "", content)
    content = re.sub(r"`+[^`\n]*`+", "", content)
    for match in re.finditer(r"!?\[[^\]\n]*\]\(([^\n]+?)\)", content):
        target = match.group(1).split(' "', 1)[0].strip("<>")
        url = urlsplit(target)
        if url.scheme or not url.path:
            continue
        link_count += 1
        resolved = (source.parent / unquote(url.path)).resolve()
        reason = ""
        if not resolved.exists():
            reason = "missing"
        elif not resolved.is_relative_to(ROOT):
            reason = "outside_git_project"
        else:
            relative = resolved.relative_to(ROOT).as_posix()
            if relative not in portable and not (
                resolved.is_dir()
                and any(p.startswith(relative + "/") for p in portable)
            ):
                reason = "ignored_not_portable"
        if reason:
            issues.append({"source": name, "target": target, "reason": reason})

coverage = (ROOT / "docs/homework_coverage.md").read_text(encoding="utf-8")
ids = re.findall(r"(?m)^\| (\d+\.\d+) \|", coverage)
expected = [
    f"{module}.{number}"
    for module, count in enumerate([5, 6, 8, 4, 6, 5], 1)
    for number in range(1, count + 1)
]
assert ids == expected, (ids, expected)

evidence = ROOT / "docs/evidence/diploma_2026-10-21"
manifest = json.loads((evidence / "evidence_manifest.json").read_text(encoding="utf-8"))
for name, entry in manifest["files"].items():
    raw = (evidence / name).read_bytes()
    assert len(raw) == entry["bytes"], name
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"], name
for path in evidence.glob("*.json"):
    json.loads(path.read_text(encoding="utf-8"))

frozen = json.loads((evidence / "final_code_manifest.json").read_text(encoding="utf-8"))
for name, expected_hash in frozen["runtime_changes_sha256"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash, name

historical = json.loads(
    (ROOT / "homework_4_2/historical/manifest.json").read_text(encoding="utf-8")
)
for entry in historical["files"]:
    original = git("show", f"{entry['commit']}:{entry['source_path']}")
    exported = (ROOT / entry["export"]).read_bytes()
    assert original == exported, entry["export"]
    assert hashlib.sha256(exported).hexdigest() == entry["sha256"]

mmd = (ROOT / "docs/final-agent-graph.mmd").read_text(encoding="utf-8")
edges = set(re.findall(r"(?m)^\s*(\w+)\s+(?:-->|-\.->)\s+(\w+);", mmd))
nodes = set(re.findall(r"(?m)^\s*(\w+)\(", mmd))
svg = ET.parse(ROOT / "docs/final-agent-graph.svg")
svg_edges = {
    (e.attrib["data-source"], e.attrib["data-target"])
    for e in svg.iter()
    if "data-source" in e.attrib
}
svg_nodes = {e.attrib["data-node"] for e in svg.iter() if "data-node" in e.attrib}
assert edges == svg_edges, edges ^ svg_edges
assert nodes == svg_nodes, nodes ^ svg_nodes
assert len(nodes - {"__start__", "__end__"}) == 7

result = {
    "checked_on": "2026-10-21",
    "task": "S0-010",
    "status": "PASS" if not issues else "FAIL",
    "scope_documents": len(DOCS),
    "relative_links_checked": link_count,
    "documents": DOCS,
    "links_check": "local path existence and portability; fragment anchors and external URLs not checked",
    "link_issues": issues,
    "homework_rows": len(ids),
    "evidence_files_hash_verified": len(manifest["files"]),
    "runtime_hashes_verified": len(frozen["runtime_changes_sha256"]),
    "historical_exports_git_bytes_verified": len(historical["files"]),
    "graph_nodes_including_terminals": len(nodes),
    "graph_edges": len(edges),
}
(ROOT / "docs/evidence/documentation_check_2026-10-21.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {k: v for k, v in result.items() if k != "documents"}, ensure_ascii=False
    )
)
documentation_link_issues = list(issues)


inventory = sorted(p for p in portable if p.startswith("docs/"))
outside = sorted(
    p
    for p in portable
    if (
        p.startswith("homework_")
        and p.endswith(".md")
        or p
        in {
            "README.md",
            "support_copilot/README.md",
            "eval/README.md",
            "tests/eval/corpus_suite/README.md",
            "tests/eval/corpus_suite/review/README.md",
        }
    )
)
markdown = sorted({p for p in inventory + outside if p.endswith(".md")})
issues, links, external = [], 0, []
for name in markdown:
    source = ROOT / name
    text = source.read_text(encoding="utf-8-sig")
    text = re.sub(r"(?ms)^```.*?^```[^\n]*", "", text)
    text = re.sub(r"`+[^`\n]*`+", "", text)
    for match in re.finditer(r"!?\[[^\]\n]*\]\(([^\n]+?)\)", text):
        target = match.group(1).split(' "', 1)[0].strip("<>")
        url = urlsplit(target)
        if url.scheme:
            external.append({"source": name, "target": target})
            continue
        if not url.path:
            continue
        links += 1
        resolved = (source.parent / unquote(url.path)).resolve()
        reason = None
        if not resolved.exists():
            reason = "missing"
        elif not resolved.is_relative_to(ROOT):
            reason = "outside_git_project"
        else:
            rel = resolved.relative_to(ROOT).as_posix()
            if rel not in portable and not (
                resolved.is_dir() and any(p.startswith(rel + "/") for p in portable)
            ):
                reason = "ignored_not_portable"
        if reason:
            issues.append({"source": name, "target": target, "reason": reason})
new = [
    "docs/technology_stack.md",
    "docs/architecture_patterns.md",
    "docs/diploma_submission_guide.md",
]
new_external = [item for item in external if item["source"] in new]
assert not new_external, new_external
frozen_diff = (
    subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "docs/evidence/diploma_2026-10-21",
            "app",
            "bot",
            "support_copilot",
            "tests",
            "pyproject.toml",
            "uv.lock",
            "compose.yaml",
            "Dockerfile",
        ],
        cwd=ROOT,
    )
    .decode()
    .splitlines()
)
assert not frozen_diff, frozen_diff
result = {
    "checked_on": "2026-10-21",
    "task": "S0-010 documentation follow-up",
    "status": "PASS" if not issues and not documentation_link_issues else "FAIL",
    "scope": "all portable docs files inventoried; all docs Markdown and homework Markdown plus subsystem guides checked for local linked paths",
    "docs_files_count": len(inventory),
    "docs_files": inventory,
    "outside_docs_markdown_count": len(outside),
    "outside_docs_markdown": outside,
    "markdown_documents_checked": len(markdown),
    "local_link_paths_checked": links,
    "link_issues": issues,
    "new_architecture_documents_external_urls": len(new_external),
    "immutable_evidence_and_runtime_diff": frozen_diff,
    "manual_review": [
        "Current architecture and technology roles checked against code and manifests.",
        "Current, optional, historical and planned paths distinguished in submission guide.",
        "17 ignored var links converted to labelled local raw paths, without exporting private data.",
        "Third-party implementation references removed from architecture, multi-agent and chunking narratives.",
    ],
    "limits": [
        "No network checks of external URLs; fragment anchors are not validated.",
        "Inline/fenced code examples are excluded from Markdown link parsing.",
        "Non-Markdown artifacts inventoried; no fresh PDF/HTML/image rendering or benchmark.",
        "Runtime/evidence integrity and graph checks are recorded separately in documentation_check_2026-10-21.json.",
        "Clean-copy startup, slide deck and demonstration video remain subsequent tasks.",
    ],
}
out = ROOT / "docs/evidence/diploma_document_audit_2026-10-21.json"
out.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            k: v
            for k, v in result.items()
            if k
            not in {"docs_files", "outside_docs_markdown", "manual_review", "limits"}
        },
        ensure_ascii=False,
        indent=2,
    )
)
raise SystemExit(1 if issues or documentation_link_issues else 0)
