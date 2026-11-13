"""search indexes on resume chunks

Revision ID: 1ca8e3d5213a
Revises: 3cb8699ffdf6
Create Date: 2026-11-13 07:05:53.124066

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1ca8e3d5213a'
down_revision: Union[str, Sequence[str], None] = '3cb8699ffdf6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # HNSW: approximate nearest-neighbour index for vector search.
    # vector_cosine_ops must match the operator the query uses (cosine_distance -> <=>).
    op.execute(
        "CREATE INDEX ix_resume_chunks_embedding_hnsw "
        "ON resume_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    # GIN: full-text index. The expression must match keyword_search exactly,
    # or Postgres will not use the index: to_tsvector('english', content).
    op.execute(
        "CREATE INDEX ix_resume_chunks_content_fts "
        "ON resume_chunks USING gin (to_tsvector('english', content))"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_resume_chunks_content_fts")
    op.execute("DROP INDEX IF EXISTS ix_resume_chunks_embedding_hnsw")
