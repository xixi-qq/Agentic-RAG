import os

from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker,AsyncSession
from dotenv import load_dotenv

load_dotenv()

ASYNC_DATABASE_URL = os.getenv('ASYNC_DATABASE_URL')

engine = create_async_engine(ASYNC_DATABASE_URL, echo=False,
                             pool_size=10, max_overflow=20)

AsyncSessionLocal = async_sessionmaker(bind=engine,
                                       class_=AsyncSession,
                                       expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e