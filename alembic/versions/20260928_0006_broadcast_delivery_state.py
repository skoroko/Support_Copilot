"""Persist broadcast leases and per-recipient delivery progress.

Revision ID: 20260928_0006
Revises: 20260910_0005
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260928_0006"
down_revision: str | None = "20260910_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("broadcast_queue", sa.Column("recipients", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")))
    op.add_column("broadcast_queue", sa.Column("invalid_recipient_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("broadcast_queue", sa.Column("lease_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("broadcast_queue", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("broadcast_queue", sa.Column("lease_seconds", sa.Integer(), nullable=False, server_default="300"))
    op.add_column("broadcast_queue", sa.Column("review_required", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    # Existing processing rows lack a recipient ledger. Delivery may have
    # happened already, so they require manual review and must not be replayed.
    op.execute("UPDATE broadcast_queue SET status = 'failed', review_required = true, updated_at = now() WHERE status = 'processing'")
    op.create_index("ix_broadcast_queue_lease_expiry", "broadcast_queue", ["status", "lease_expires_at"])


def downgrade() -> None:
    op.execute("UPDATE broadcast_queue SET status = 'failed', updated_at = now() WHERE status = 'processing'")
    op.drop_index("ix_broadcast_queue_lease_expiry", table_name="broadcast_queue")
    op.drop_column("broadcast_queue", "review_required")
    op.drop_column("broadcast_queue", "lease_seconds")
    op.drop_column("broadcast_queue", "lease_expires_at")
    op.drop_column("broadcast_queue", "lease_id")
    op.drop_column("broadcast_queue", "invalid_recipient_count")
    op.drop_column("broadcast_queue", "recipients")
