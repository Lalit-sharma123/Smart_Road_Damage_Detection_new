from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config.config import settings

# Create Async Engine for PostgreSQL
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)

# Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)


class Base(DeclarativeBase):
    """Base ORM declarative class"""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Dependency to yield Async Database Sessions.
    Guarantees session cleanup upon request termination.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create database tables if they do not exist"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
