from typing import Optional
from sqlalchemy import Integer, String, ForeignKey, Text, Enum, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from models.base import Base


class Document(Base):
    __tablename__ = "documents"

    __table_args__ = (
        UniqueConstraint("user_id", "file_hash",name='uq_user_filehash'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id",ondelete="CASCADE"),
                                         nullable=False,index=True,)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(Enum('pending', 'processing', 'completed', 'failed',
                                             name="document_status",native_enum=False),
                                        default='pending', nullable=False)
    file_hash: Mapped[str] = mapped_column(String(100), nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer,default=0, nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bucket_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index",name='uq_document_chunk_index'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id",ondelete="CASCADE"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    vector_id: Mapped[Optional[str]] = mapped_column(String(100),nullable=True)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)



