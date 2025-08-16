import logging
from typing import Dict, List

logger = logging.getLogger("eora")


class ContextBuilder:
    """Класс для создания контекста из результатов поиска."""

    def __init__(self, max_context_length: int = 32):
        """
        Args:
            max_context_length: Максимальное количество фрагментов в контексте
        """
        self.max_context_length = max_context_length

    def create_context(self, search_results: List[Dict], question: str) -> str:
        """
        Создает улучшенный контекст с приоритизацией и фильтрацией.

        Args:
            search_results: Результаты поиска
            question: Вопрос пользователя
        """
        if not search_results:
            return ""

        # Группируем результаты по источникам для лучшей организации
        by_source = {}
        for result in search_results:
            source = result["source"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result)

        context_parts = []

        # Добавляем результаты, организованные по источникам
        for source, results in by_source.items():
            # Берем лучшие результаты из каждого источника
            best_results = sorted(results, key=lambda x: x["similarity"], reverse=True)[
                :10
            ]

            for result in best_results:
                if result["similarity"] > 0.2:  # Фильтруем слишком слабые совпадения
                    context_parts.append(
                        f"Источник: {result['source']}\n"
                        f"Релевантность: {result['similarity']:.2f}\n"
                        f"Контент: {result['text']}\n"
                    )

        return "\n---\n".join(context_parts[: self.max_context_length])

    def extract_sources(self, search_results: List[Dict], limit: int = 5) -> List[Dict]:
        """
        Извлекает и форматирует источники из результатов поиска.

        Args:
            search_results: Результаты поиска
            limit: Максимальное количество источников
        """
        all_sources = set()
        for res in search_results:
            all_sources.add(res["source"])
        sources = sorted(list(all_sources))[:limit]
        formatted_sources = []

        for source in sources:
            # Извлекаем название проекта из URL
            project_name = source.split("/")[-1].replace("-", " ").title()
            formatted_sources.append({"name": project_name, "url": source})

        return formatted_sources
