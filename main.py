import asyncio
import os
from parser import ParserConfig

from ai import AIClient, ContextBuilder, SearchEngine, VectorDatabaseManager

from logger import logger
from telegram_bot import TelegramBot
from config import *

# Получаем токен из переменной окружения или запрашиваем у пользователя
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    TELEGRAM_TOKEN = input("Введите токен Telegram бота: ")
    if not TELEGRAM_TOKEN:
        raise ValueError("Не указан токен Telegram бота")


async def main():
    """Основная функция для запуска Telegram бота."""
    try:
        ai_client = AIClient(
            api_key=OPENAI_API_KEY,
            api_base_url=API_BASE_URL,
            embedding_model=EMBEDDING_MODEL,
        )

        db_manager = VectorDatabaseManager(
            index_dir=INDEX_DIR,
            hnsw_index_path=HNSW_INDEX_PATH,
            metadata_path=METADATA_PATH,
            embeddings_dim=EMBEDDINGS_DIM,
        )

        parser_config = ParserConfig(
            api_base_url=API_BASE_URL,
            openai_api_key=OPENAI_API_KEY,
            embedding_model=EMBEDDING_MODEL,
            embeddings_dim=EMBEDDINGS_DIM,
            index_dir=INDEX_DIR,
            hnsw_index_path=HNSW_INDEX_PATH,
            metadata_path=METADATA_PATH,
            request_headers=REQUEST_HEADERS,
            urls_to_parse=URLS_TO_PARSE,
        )

        logger.info("Загрузка векторной базы данных...")
        index, metadata = await db_manager.create_or_load_database(parser_config)

        if not index or not metadata:
            logger.error(
                "Не удалось загрузить или создать базу данных. Прерываю выполнение."
            )
            return

        search_engine = SearchEngine(ai_client, index, metadata)
        context_builder = ContextBuilder(max_context_length=32)
        bot = TelegramBot(
            token=TELEGRAM_TOKEN,
            search_engine=search_engine,
            context_builder=context_builder,
            ai_client=ai_client,
        )

        # Запускаем бота
        await bot.setup()

        # Ждем прерывания
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            await bot.shutdown()

    except Exception as e:
        logger.error(f"Критическая ошибка при запуске бота: {e}")
        print(f"❌ Критическая ошибка при запуске бота: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен пользователем.")
