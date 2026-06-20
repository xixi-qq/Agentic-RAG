from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.users import User
from apps.user.schemas import UserRegister
from utils.security import hash_password


async def get_user_by_username(username: str, db: AsyncSession):
    stmt = select(User).where(User.name == username)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_by_phone(phone: str, db: AsyncSession):
    stmt = select(User).where(User.phone == phone)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def create_user(user_data: UserRegister, db: AsyncSession):
    hash_pwd = hash_password(user_data.password)
    user = User(name=user_data.name, password=hash_pwd, phone=user_data.phone)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user