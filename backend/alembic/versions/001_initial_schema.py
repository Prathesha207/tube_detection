from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial_schema'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table(
        'cameras',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('ip_address', sa.String(length=255), nullable=True),
        sa.Column('stream_url', sa.String(length=255), nullable=True),
        sa.Column('is_enabled', sa.Boolean(), default=True),
        sa.Column('recording_video_path', sa.String(), nullable=True),
        sa.Column('recording_video_testing_path', sa.String(), nullable=True),
        sa.Column('resolution', sa.String(), nullable=True),
        sa.Column('fps', sa.Integer(), nullable=True),
        sa.Column('device_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_table(
        'recordings',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('camera_id', sa.Integer(), sa.ForeignKey('cameras.id'), nullable=True),
        sa.Column('file_path', sa.String(length=255), nullable=False),
        sa.Column('start_time', sa.DateTime(), nullable=True),
        sa.Column('end_time', sa.DateTime(), nullable=True),
    )

def downgrade() -> None:
    op.drop_table('recordings')
    op.drop_table('cameras')
