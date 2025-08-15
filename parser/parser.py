import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import hnswlib

from .embedding_manager import EmbeddingManager
from logger import logger
from .text_processor import TextProcessor
from .vector_builder import VectorDBBuilder
from .web_scrapper import WebScraper


@dataclass
class ParserConfig:
    """Конфигурация парсера."""

    openai_api_key: str
    embedding_model: str
    urls_to_parse: List[str]
    api_base_url: str = "https://api.openai.com/v1/"
    embeddings_dim: int = 1536
    index_dir: str = "vector_storage"
    metadata_path: str = "vector_storage/metadata.json"
    hnsw_index_path: str = "vector_storage/hnsw_index.bin"
    request_headers: Dict[str, str] = dict


class Parser:
    """Основной класс для парсинга веб-страниц и создания векторной базы данных."""

    def __init__(self, config: ParserConfig):
        """
        Инициализация класса Parser.

        Args:
            config: Конфигурация парсера
        """
        # Проверка входных данных
        self._validate_config(config)
        self.config = config

        # Инициализация компонентов
        self.scraper = WebScraper(config.request_headers)
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager(
            config.api_base_url,
            config.openai_api_key,
            config.embedding_model,
            config.embeddings_dim,
            config.index_dir,
        )
        self.vector_db_builder = VectorDBBuilder(
            config.hnsw_index_path, config.metadata_path, config.embeddings_dim
        )

        # Статистика выполнения
        self.stats = {
            "urls_processed": 0,
            "urls_failed": 0,
            "chunks_created": 0,
            "start_time": time.time(),
            "total_time": 0.0,
        }

    def _validate_config(self, config: ParserConfig) -> None:
        """
        Проверяет конфигурацию на корректность.

        Args:
            config: Конфигурация для проверки
        """
        if not config.api_base_url:
            raise ValueError("API base URL не может быть пустым")
        if not config.openai_api_key:
            raise ValueError("OpenAI API key не может быть пустым")
        if not config.embedding_model:
            raise ValueError("Модель эмбеддингов не может быть пустой")
        if config.embeddings_dim <= 0:
            raise ValueError("Размерность эмбеддингов должна быть положительной")
        if not config.urls_to_parse:
            raise ValueError("Список URL для парсинга не может быть пустым")

    async def _parse_and_create_vector_db(
        self,
    ) -> Tuple[Optional[hnswlib.Index], List[Dict[str, str]]]:
        """
        Асинхронно собирает данные, затем создает и сохраняет векторную базу.

        Returns:
            Кортеж (индекс HNSW, метаданные)
        """
        start_time = time.time()
        logger.info("Начинаю процесс создания векторной базы...")

        # Парсинг страниц
        logger.info(f"Парсинг {len(self.config.urls_to_parse)} страниц...")
        scraped_docs = await self.scraper.scrape_urls(self.config.urls_to_parse)

        self.stats["urls_processed"] = len(scraped_docs)
        self.stats["urls_failed"] = len(self.config.urls_to_parse) - len(scraped_docs)

        if not scraped_docs:
            logger.error("Не удалось спарсить ни одну страницу. Прерываю выполнение.")
            return None, []

        # Разделение текстов на чанки
        all_chunks = await self.text_processor.process_documents(scraped_docs)
        self.stats["chunks_created"] = len(all_chunks)

        if not all_chunks:
            logger.error("Не создано ни одного чанка. Прерываю выполнение.")
            return None, []

        # Получение эмбеддингов
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = await self.embedding_manager.get_embeddings(chunk_texts)

        # Создание и сохранение индекса
        index, metadata = self.vector_db_builder.create_index(embeddings, all_chunks)

        # Обновление статистики
        self.stats["total_time"] = time.time() - start_time
        self.stats["embedding_stats"] = self.embedding_manager.stats

        logger.info(
            f"Векторная база успешно создана за {self.stats['total_time']:.2f} секунд!"
        )
        logger.info(
            f"Обработано URL: {self.stats['urls_processed']}/{len(self.config.urls_to_parse)}"
        )
        logger.info(f"Создано чанков: {self.stats['chunks_created']}")

        return index, metadata

    async def run(self) -> Dict[str, Any]:
        """
        Запускает процесс парсинга и создания векторной базы.

        Returns:
            Dict: Статистика выполнения
        """
        await self._parse_and_create_vector_db()
        return self.stats

    def load_vector_db(self) -> Tuple[Optional[hnswlib.Index], List[Dict[str, str]]]:
        """
        Загружает сохраненную векторную базу данных.

        Returns:
            Кортеж (индекс HNSW, метаданные)
        """
        return self.vector_db_builder.load_index()
