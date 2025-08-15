import asyncio
import gzip
import hashlib
import json
import os
import time
from typing import Dict, List

import httpx
import numpy as np
from tqdm import tqdm

from logger import logger


class EmbeddingManager:
    """Класс для управления эмбеддингами, включая кэширование."""

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model: str,
        embeddings_dim: int,
        cache_dir: str,
    ):
        """
        Инициализация EmbeddingManager.

        Args:
            api_base_url: Базовый URL для API
            api_key: API ключ
            model: Модель эмбеддингов
            embeddings_dim: Размерность эмбеддингов
            cache_dir: Директория для хранения кэша
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model = model
        self.embeddings_dim = embeddings_dim
        self.cache_path = os.path.join(cache_dir, "embedding_cache.json.gz")

        # Статистика для мониторинга
        self.stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
        }

    def _load_cache(self) -> Dict[str, List[float]]:
        """
        Загружает кэш эмбеддингов из файла.

        Returns:
            Словарь с кэшированными эмбеддингами
        """
        try:
            with gzip.open(self.cache_path, "rt") as f:
                cache_data = json.load(f)
                cache = cache_data.get("embeddings", {})
                cache_model = cache_data.get("model", self.model)

                # Если модель изменилась, очищаем кэш
                if cache_model != self.model:
                    logger.warning(
                        f"Модель эмбеддингов изменилась ({cache_model} -> {self.model}). Кэш будет сброшен."
                    )
                    return {}

                return cache
        except (FileNotFoundError, json.JSONDecodeError, gzip.BadGzipFile):
            return {}

    def _save_cache(self, cache: Dict[str, List[float]]) -> None:
        """
        Сохраняет кэш эмбеддингов в файл.

        Args:
            cache: Словарь с кэшированными эмбеддингами
        """
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        try:
            with gzip.open(self.cache_path, "wt") as f:
                json.dump(
                    {
                        "version": "1.1",
                        "model": self.model,
                        "created_at": time.time(),
                        "embeddings": cache,
                    },
                    f,
                )
        except Exception as e:
            logger.error(f"Ошибка при сохранении кэша эмбеддингов: {e}")

    async def get_embeddings(
        self, texts: List[str], batch_size: int = 32, max_retries: int = 5
    ) -> np.ndarray:
        """
        Получает эмбеддинги с использованием кэша.

        Args:
            texts: Список текстов для получения эмбеддингов
            batch_size: Размер пакета текстов для одного запроса
            max_retries: Максимальное количество попыток при ошибках

        Returns:
            Массив эмбеддингов
        """
        # Загружаем кэш
        cache = self._load_cache()

        # Подготавливаем данные
        final_embeddings = [None] * len(texts)
        original_indices = {}

        # Проверяем кэш
        for i, text in enumerate(texts):
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if text_hash in cache:
                final_embeddings[i] = cache[text_hash]
                self.stats["cache_hits"] += 1
            else:
                if text not in original_indices:
                    original_indices[text] = []
                original_indices[text].append(i)
                self.stats["cache_misses"] += 1

        texts_to_fetch = list(original_indices.keys())

        logger.info(
            f"Найдено в кэше: {len(texts) - len(texts_to_fetch)}/{len(texts)} эмбеддингов."
        )

        # Получаем недостающие эмбеддинги
        if texts_to_fetch:
            new_embeddings = await self._fetch_embeddings(
                texts_to_fetch, batch_size, max_retries
            )

            for i, text in enumerate(texts_to_fetch):
                embedding = new_embeddings[i].tolist()
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                cache[text_hash] = embedding

                for original_index in original_indices[text]:
                    final_embeddings[original_index] = embedding

        # Сохраняем обновленный кэш
        self._save_cache(cache)

        return np.array(final_embeddings, dtype=np.float32)

    async def _fetch_embeddings(
        self, texts: List[str], batch_size: int = 32, max_retries: int = 5
    ) -> np.ndarray:
        """
        Асинхронно получает эмбеддинги для списка текстов.

        Args:
            texts: Список текстов для получения эмбеддингов
            batch_size: Размер пакета текстов для одного запроса
            max_retries: Максимальное количество попыток при ошибках

        Returns:
            Массив эмбеддингов
        """
        all_embeddings = []

        async with httpx.AsyncClient(timeout=60) as client:
            for i in tqdm(
                range(0, len(texts), batch_size), desc="Получение эмбеддингов"
            ):
                batch = texts[i : i + batch_size]
                if not batch:
                    continue

                self.stats["api_calls"] += 1
                self.stats["total_tokens"] += sum(len(text.split()) for text in batch)

                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            f"{self.api_base_url}embeddings",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json={"input": batch, "model": self.model},
                        )
                        response.raise_for_status()
                        data = response.json()
                        batch_embeddings = sorted(
                            data["data"], key=lambda e: e["index"]
                        )
                        all_embeddings.extend(
                            [item["embedding"] for item in batch_embeddings]
                        )

                        # Экспоненциальная задержка между пакетами для защиты от rate limits
                        await asyncio.sleep(0.5 * (2 ** min(attempt, 3)))
                        break
                    except httpx.RequestError as e:
                        wait_time = 2**attempt
                        logger.warning(
                            f"Ошибка при получении эмбеддингов: {e}. "
                            f"Попытка {attempt + 1}/{max_retries}. "
                            f"Повтор через {wait_time} сек."
                        )
                        if attempt + 1 == max_retries:
                            logger.error(
                                f"Не удалось обработать пакет после {max_retries} попыток."
                            )
                            all_embeddings.extend(
                                [[0.0] * self.embeddings_dim] * len(batch)
                            )
                        else:
                            await asyncio.sleep(wait_time)

        return np.array(all_embeddings, dtype=np.float32)
