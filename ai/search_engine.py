import asyncio
import logging
import re
from typing import Dict, List, Set

import hnswlib

logger = logging.getLogger("eora")


class SearchEngine:
    """Класс для поиска информации в векторной базе данных."""

    def __init__(self, ai_client, index: hnswlib.Index, metadata: List[Dict]):
        """
        Args:
            ai_client: Экземпляр класса AIClient для получения эмбеддингов
            index: Индекс HNSW для поиска ближайших соседей
            metadata: Метаданные документов
        """
        self.ai_client = ai_client
        self.index = index
        self.metadata = metadata
        self.stats = {
            "total_searches": 0,
            "direct_hits": 0,
            "expanded_hits": 0,
            "keyword_hits": 0,
        }

    def expand_query(self, question: str) -> List[str]:
        """
        Расширяет запрос синонимами и альтернативными формулировками.

        Args:
            question: Исходный вопрос
        """
        synonyms = {
            "проект": ["кейс", "задача", "решение", "разработка"],
            "алгоритм": ["нейросеть", "модель", "ИИ", "AI", "машинное обучение"],
            "бот": ["чат-бот", "ассистент", "помощник", "chatbot"],
            "компания": ["клиент", "заказчик", "организация", "фирма"],
            "технология": ["решение", "система", "платформа", "инструмент"],
            "анализ": ["обработка", "исследование", "изучение", "аналитика"],
            "автоматизация": ["оптимизация", "упрощение", "ускорение"],
        }

        expanded_queries = [question]

        for word, syns in synonyms.items():
            if word in question.lower():
                for syn in syns:
                    expanded_queries.append(question.lower().replace(word, syn))

        return expanded_queries

    def extract_keywords(self, text: str) -> List[str]:
        """
        Извлекает ключевые слова из текста.

        Args:
            text: Исходный текст
        """
        words = re.findall(r"\b[а-яёА-ЯЁa-zA-Z]{3,}\b", text.lower())
        # Убираем слова, не несущие смысловой нагрузки
        stop_words = {
            "как",
            "что",
            "где",
            "когда",
            "почему",
            "который",
            "которая",
            "которое",
            "для",
            "или",
            "это",
            "есть",
            "был",
            "была",
            "было",
            "были",
            "может",
        }
        keywords = [word for word in words if word not in stop_words]
        return list(set(keywords))

    def keyword_search(self, keywords: List[str], threshold: int = 1) -> List[Dict]:
        """
        Простой поиск по ключевым словам.

        Args:
            keywords: Список ключевых слов
            threshold: Минимальное количество совпадений для включения в результаты
        """
        results = []

        for i, chunk in enumerate(self.metadata):
            text_lower = chunk["text"].lower()
            matches = sum(1 for keyword in keywords if keyword in text_lower)

            if matches >= threshold:
                results.append(
                    {
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "similarity": matches / len(keywords),
                        "strategy": "keyword",
                    }
                )

        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]

    async def search(self, question: str, top_k: int = 10) -> List[Dict]:
        """
        Выполняет поиск с использованием нескольких стратегий.

        Args:
            question: Вопрос пользователя
            top_k: Количество результатов для возврата
        """
        self.stats["total_searches"] += 1
        all_results = []
        seen_texts: Set[str] = set()

        # Стратегия 1: Прямой поиск по исходному вопросу
        query_vector = await self.ai_client.get_embedding(question)
        if query_vector is not None:
            labels, distances = await asyncio.to_thread(
                self.index.knn_query, query_vector, k=top_k
            )

            for label, dist in zip(labels[0], distances[0]):
                text = self.metadata[label]["text"]
                if text not in seen_texts:
                    all_results.append(
                        {
                            "text": text,
                            "source": self.metadata[label]["source"],
                            "similarity": 1 - dist,
                            "strategy": "direct",
                        }
                    )
                    seen_texts.add(text)
                    self.stats["direct_hits"] += 1

        # Стратегия 2: Поиск по расширенным запросам
        expanded_queries = self.expand_query(question)
        expanded_embedding_tasks = []

        for expanded_query in expanded_queries[1:3]:  # Берем только первые 2 расширения
            if expanded_query != question:
                expanded_embedding_tasks.append(
                    self.ai_client.get_embedding(expanded_query)
                )

        if expanded_embedding_tasks:  # Проверка на пустой список
            expanded_embeddings = await asyncio.gather(*expanded_embedding_tasks)

            for i, exp_vector in enumerate(expanded_embeddings):
                if exp_vector is not None:
                    labels, distances = await asyncio.to_thread(
                        self.index.knn_query, exp_vector, k=5
                    )

                    for label, dist in zip(labels[0], distances[0]):
                        text = self.metadata[label]["text"]
                        if text not in seen_texts and (1 - dist) > 0.6:
                            all_results.append(
                                {
                                    "text": text,
                                    "source": self.metadata[label]["source"],
                                    "similarity": 1 - dist,
                                    "strategy": "expanded",
                                }
                            )
                            seen_texts.add(text)
                            self.stats["expanded_hits"] += 1

        # Стратегия 3: Keyword-based поиск для fallback
        keywords = self.extract_keywords(question)
        if keywords:
            keyword_results = await asyncio.to_thread(self.keyword_search, keywords)

            for result in keyword_results:
                if result["text"] not in seen_texts:
                    all_results.append(result)
                    seen_texts.add(result["text"])
                    self.stats["keyword_hits"] += 1

        # Сортируем по схожести и берем топ результаты
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    def get_stats(self) -> Dict:
        return self.stats
