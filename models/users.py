from typing import Optional
from sqlalchemy import Integer, String, Index,Enum
from sqlalchemy.orm import Mapped, mapped_column
from models.base import Base


class User(Base):
    __tablename__ = "users"

    __table_args__ = (
        Index("idx_users_name", "name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True,autoincrement=True,)
    name: Mapped[str] = mapped_column(String(20),unique=True,nullable=False)
    password: Mapped[str] = mapped_column(String(255),nullable=False)
    phone: Mapped[str] = mapped_column(String(11),unique=True,nullable=False)
    gender: Mapped[Optional[str]] = mapped_column(Enum('male', 'female', 'unknown',native_enum=False),
                                                  default='unknown', nullable=False)