import asyncio
import re
from typing import Dict, List

import nltk

from logger import logger


class TextProcessor:
    """Класс для обработки и разделения текста на чанки."""

    def __init__(self):
        """Инициализация TextProcessor."""
        self._init_nltk()

    def _init_nltk(self):
        """Инициализация NLTK для токенизации предложений."""
        try:
            nltk.data.find("tokenizers/punkt/russian.pickle")
        except LookupError:
            logger.info("Загрузка пакета 'punkt' для NLTK...")
            nltk.download("punkt")

    def split_text(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """
        Разделяет текст на чанки, сохраняя целостность предложений.

        Args:
            text: Текст для разделения
            chunk_size: Максимальный размер чанка в символах
            chunk_overlap: Размер перекрытия между чанками в символах

        Returns:
            Список чанков текста
        """
        if not text:
            return []

        # Удаляем избыточные пробелы и переносы строк
        text = re.sub(r"\s+", " ", text).strip()

        # Разделяем текст на предложения
        sentences = nltk.sent_tokenize(text, language="russian")

        # Объединяем предложения в чанки с учетом перекрытия
        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            # Если текущий чанк пуст или добавление предложения не превысит лимит
            if not current_chunk or current_length + len(sentence) <= chunk_size:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                # Сохраняем текущий чанк и начинаем новый
                chunks.append(" ".join(current_chunk))

                # Находим точку для перекрытия (backtracking)
                overlap_end = 0
                overlap_length = 0

                # Идем с конца текущего чанка, добавляя предложения, пока не достигнем
                # желаемого размера перекрытия
                for j in range(len(current_chunk) - 1, -1, -1):
                    overlap_length += len(current_chunk[j])
                    if overlap_length >= chunk_overlap:
                        overlap_end = j
                        break

                # Новый чанк начинается с предложений перекрытия плюс текущее предложение
                current_chunk = current_chunk[overlap_end:] + [sentence]
                current_length = sum(len(s) for s in current_chunk)

        # Добавляем последний чанк, если он не пуст
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def process_documents(
        self,
        documents: List[Dict[str, str]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Dict[str, str]]:
        """
        Обрабатывает список документов и разделяет их на чанки.

        Args:
            documents: Список документов для обработки
            chunk_size: Максимальный размер чанка
            chunk_overlap: Размер перекрытия между чанками

        Returns:
            Список чанков с метаданными
        """
        all_chunks = []

        async def process_single_document(doc):
            chunks = await asyncio.to_thread(
                self.split_text, doc["page_content"], chunk_size, chunk_overlap
            )
            return [
                {"text": chunk_text, "source": doc["source"]} for chunk_text in chunks
            ]

        # Запускаем обработку всех документов параллельно
        tasks = [process_single_document(doc) for doc in documents]
        chunks_per_doc = await asyncio.gather(*tasks)

        # Объединяем результаты
        for doc_chunks in chunks_per_doc:
            all_chunks.extend(doc_chunks)

        logger.info(f"Тексты разделены на {len(all_chunks)} чанков.")
        return all_chunks
