
import asyncio

from config.qdrant_config import client, ensure_collection


async def main():
    try:
        await ensure_collection()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())