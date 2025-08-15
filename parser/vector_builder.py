import json
import os
from typing import Dict, List, Optional, Tuple

import hnswlib
import numpy as np

from logger import logger


class VectorDBBuilder:
    """Класс для создания и управления векторной базой данных."""

    def __init__(self, index_path: str, metadata_path: str, embeddings_dim: int):
        """
        Инициализация VectorDBBuilder.

        Args:
            index_path: Путь к файлу индекса
            metadata_path: Путь к файлу метаданных
            embeddings_dim: Размерность эмбеддингов
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embeddings_dim = embeddings_dim

    def create_index(
        self, embeddings: np.ndarray, chunks: List[Dict[str, str]]
    ) -> Tuple[hnswlib.Index, List[Dict[str, str]]]:
        """
        Создает индекс HNSW из эмбеддингов и сохраняет его.

        Args:
            embeddings: Массив эмбеддингов
            chunks: Список чанков с метаданными

        Returns:
            Кортеж (индекс HNSW, метаданные)
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Количество эмбеддингов ({embeddings.shape[0]}) не совпадает с количеством чанков ({len(chunks)})"
            )

        # Создаем и наполняем индекс
        logger.info("Создание HNSWLib индекса...")
        index = hnswlib.Index(space="cosine", dim=self.embeddings_dim)
        index.init_index(max_elements=len(chunks), ef_construction=200, M=16)
        index.add_items(embeddings, np.arange(len(chunks)))

        # Сохраняем индекс и метаданные
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        logger.info(f"Сохранение индекса в файл: {self.index_path}")
        index.save_index(self.index_path)

        logger.info(f"Сохранение метаданных в файл: {self.metadata_path}")
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return index, chunks

    def load_index(self) -> Tuple[Optional[hnswlib.Index], List[Dict[str, str]]]:
        """
        Загружает сохраненную векторную базу данных.

        Returns:
            Кортеж (индекс HNSW, метаданные)
        """
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            index = hnswlib.Index(space="cosine", dim=self.embeddings_dim)
            index.load_index(self.index_path, max_elements=len(metadata))

            logger.info(
                f"База данных успешно загружена. Количество документов: {index.get_current_count()}"
            )
            return index, metadata
        except FileNotFoundError:
            logger.error("Файлы индекса или метаданных не найдены!")
            logger.error(
                f"Убедитесь, что файлы '{self.index_path}' и '{self.metadata_path}' существуют."
            )
            return None, []
