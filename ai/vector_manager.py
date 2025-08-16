import json
import logging
import os
from parser import Parser, ParserConfig
from typing import Dict, List, Optional, Tuple

import hnswlib

logger = logging.getLogger("eora")


class VectorDatabaseManager:
    """Класс для управления векторной базой данных."""

    def __init__(
        self,
        index_dir: str,
        hnsw_index_path: str,
        metadata_path: str,
        embeddings_dim: int,
    ):
        """
        Инициализация менеджера векторной базы данных.

        Args:
            index_dir: Директория для хранения индексов
            hnsw_index_path: Путь к файлу индекса
            metadata_path: Путь к файлу метаданных
            embeddings_dim: Размерность эмбеддингов
        """
        self.index_dir = index_dir
        self.hnsw_index_path = hnsw_index_path
        self.metadata_path = metadata_path
        self.embeddings_dim = embeddings_dim

        # Создаем директорию, если она не существует
        os.makedirs(index_dir, exist_ok=True)

    async def create_or_load_database(
        self, parser_config: ParserConfig
    ) -> Tuple[Optional[hnswlib.Index], List[Dict[str, str]]]:
        """
        Создает или загружает векторную базу данных.

        Args:
            parser_config: Конфигурация для парсера

        Returns:
            Кортеж (индекс HNSW, метаданные)
        """
        # Проверяем, существуют ли уже файлы индекса и метаданных
        if os.path.exists(self.hnsw_index_path) and os.path.exists(self.metadata_path):
            logger.info("Найдены существующие файлы индекса и метаданных. Загружаем...")
            return self.load_database()
        else:
            logger.info("Файлы индекса не найдены. Создаем новую базу данных...")
            return await self.create_database(parser_config)

    async def create_database(
        self, parser_config: ParserConfig
    ) -> Tuple[Optional[hnswlib.Index], List[Dict[str, str]]]:
        """
        Создает новую векторную базу данных.

        Args:
            parser_config: Конфигурация для парсера

        Returns:
            Кортеж (индекс HNSW, метаданные)
        """
        parser = Parser(parser_config)
        stats = await parser.run()

        logger.info(
            f"Парсинг и создание базы завершены за {stats['total_time']:.2f} секунд"
        )
        logger.info(
            f"Обработано URL: {stats['urls_processed']}/{len(parser_config.urls_to_parse)}"
        )
        logger.info(f"Создано чанков: {stats['chunks_created']}")

        return parser.load_vector_db()

    def load_database(self) -> Tuple[Optional[hnswlib.Index], List[Dict[str, str]]]:
        """
        Загружает существующую векторную базу данных.

        Returns:
            Кортеж (индекс HNSW, метаданные)
        """
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            index = hnswlib.Index(space="cosine", dim=self.embeddings_dim)
            index.load_index(self.hnsw_index_path, max_elements=len(metadata))

            logger.info(
                f"База данных успешно загружена. Количество документов: {len(metadata)}"
            )
            return index, metadata
        except Exception as e:
            logger.error(f"Ошибка при загрузке базы данных: {e}")
            return None, []
