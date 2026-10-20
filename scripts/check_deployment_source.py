"""Read-only comparison of known application files with running Compose images.

Only file hashes and selected image metadata leave the container. No environment
values, provider clients, raw Docker configuration or application logs are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "app/services/agent_persistent.py", "app/services/rag.py",
    "app/services/evidence_text.py", "app/services/evidence_scope.py",
    "app/services/overview_evidence.py", "app/services/verified_overview.py",
    "app/services/runtime_model.py", "app/chat/agent.py", "app/core/config.py",
    "app/services/turn_budget.py", "app/services/agent_history.py",
    "app/services/agent_retrieval.py",
    "app/services/provider_race.py",
    "app/services/prepared_evidence.py",
    "app/services/agent_evidence_input.py",
    "app/services/literal_location.py", "app/services/related_evidence.py",
    "app/services/retrieval_embedding.py", "app/services/retrieval_translation.py",
    "app/services/retrieval_planner.py",
    "app/services/parallel_retrieval.py", "app/chat/routes.py", "app/main.py",
    "bot/services/backend_client.py",
    "bot/handlers/text.py", "bot/handlers/expansion.py",
)


def command(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
    return result.stdout.strip()


def inspect_sources() -> dict:
    expected = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in FILES}
    # The probe contains only a literal whitelist of source paths, never a scan.
    probe = (
        "import hashlib,json; from pathlib import Path; "
        f"paths={FILES!r}; "
        "print(json.dumps({p:hashlib.sha256(Path('/app',p).read_bytes()).hexdigest() "
        "if Path('/app',p).is_file() else None for p in paths}))"
    )
    services = []
    for service in ("app", "bot"):
        row: dict = {"service": service}
        try:
            container = command("docker", "compose", "ps", "-q", service)
            if not re.fullmatch(r"[0-9a-f]{12,64}", container):
                row["status"] = "not_running"
                services.append(row)
                continue
            image = command("docker", "inspect", "--format", "{{.Image}}", container)
            revision = command("docker", "inspect", "--format",
                               '{{ index .Config.Labels "org.opencontainers.image.revision" }}', container)
            actual = json.loads(command("docker", "exec", container, "python", "-c", probe))
            matches = {path: actual.get(path) == digest for path, digest in expected.items()}
            row.update(status="matches_working_tree" if all(matches.values()) else "source_mismatch",
                       image_id=image if re.fullmatch(r"sha256:[0-9a-f]{64}", image) else "unknown",
                       revision=revision if re.fullmatch(r"[0-9a-f]{40,64}", revision) else None,
                       files={path: {"matches": matches[path], "expected_sha256": digest,
                                     "deployed_sha256": actual.get(path)} for path, digest in expected.items()})
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            # stderr may contain connection/configuration details; retain only type.
            row.update(status="inspection_unavailable", error_type=type(exc).__name__)
        services.append(row)
    return {"captured_at": datetime.now(UTC).isoformat(), "read_only": True,
            "comparison": "exact bytes in current working tree, including uncommitted changes",
            "services": services}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_sources()
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(json.dumps({"read_only": True, "services": [
        {key: row[key] for key in ("service", "status", "image_id", "revision", "error_type") if key in row}
        | {"mismatched_files": [path for path, result in row.get("files", {}).items() if not result["matches"]]}
        for row in report["services"]]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(row["status"] == "matches_working_tree" for row in report["services"]) else 1)
