import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger("eora")


class AIClient:
    """
    Класс для работы с API языковых моделей и эмбеддингов.
    """

    def __init__(self, api_key: str, api_base_url: str, embedding_model: str):
        """
        Инициализация клиента AI.

        Args:
            api_key: API ключ для доступа к сервисам
            api_base_url: Базовый URL для API запросов
            embedding_model: Модель для создания эмбеддингов
        """
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.embedding_model = embedding_model
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)

        # Системный промпт по умолчанию
        self.default_system_prompt = """Ты — экспертный консультант компании Eora, которая специализируется на разработке AI решений, чат-ботов, нейросетей и автоматизации бизнес-процессов.

ТВОЯ РОЛЬ:
- Отвечай от имени компании (используй "мы", "наша компания")
- Ты эксперт в области искусственного интеллекта и автоматизации
- Давай конкретные, практические и полезные ответы

ИНСТРУКЦИИ ПО ОТВЕТАМ:
1. Используй предоставленный контекст как основной источник информации, но не упоминай наличие этого контекста в диалоге.
2. Если в контексте есть релевантная информация - отвечай подробно и конкретно
3. Если контекст частично релевантен - используй его и дополни общими знаниями о AI/ML
4. Приводи примеры из наших проектов, если они есть в контексте
5. Будь конкретным: называй технологии, подходы, результаты
6. Если не можешь ответить точно - честно скажи об этом ("К сожалению, у нас нет подобного опыта", например), но предложи альтернативы
7. Используй простое форматирование текста - избегай заголовков разных уровней, лучше использовать жирный текст и курсв так, чтобы он корректно отображался в Telegram.

СТИЛЬ ОТВЕТА:
- Профессиональный, но понятный
- Структурированный (используй списки, заголовки при необходимости)
- Включай практические советы и рекомендации
"""

        # Статистика для мониторинга использования API
        self.stats = {
            "embedding_calls": 0,
            "completion_calls": 0,
            "tokens_used": 0,
        }

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Получает эмбеддинг для текста.

        Args:
            text: Текст для эмбеддинга

        Returns:
            Эмбеддинг в виде numpy массива или None в случае ошибки
        """
        try:
            self.stats["embedding_calls"] += 1
            response = await self.client.embeddings.create(
                input=[text], model=self.embedding_model
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга: {e}")
            return None

    async def get_batch_embeddings(
        self, texts: List[str], batch_size: int = 32, max_retries: int = 3
    ) -> np.ndarray:
        """
        Получает эмбеддинги для списка текстов, обрабатывая их пакетами.

        Args:
            texts: Список текстов для эмбеддингов
            batch_size: Размер пакета для обработки
            max_retries: Максимальное количество повторных попыток при ошибке

        Returns:
            Массив эмбеддингов
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if not batch:
                continue

            self.stats["embedding_calls"] += 1

            for attempt in range(max_retries):
                try:
                    response = await self.client.embeddings.create(
                        input=batch, model=self.embedding_model
                    )

                    # Сортируем по индексу, чтобы сохранить порядок
                    batch_embeddings = sorted(response.data, key=lambda e: e.index)
                    all_embeddings.extend([item.embedding for item in batch_embeddings])

                    # Пауза между запросами для соблюдения rate limits
                    await asyncio.sleep(0.5)
                    break
                except Exception as e:
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
                        # Возвращаем нулевые эмбеддинги как fallback
                        all_embeddings.extend([[0.0] * 1536] * len(batch))
                    else:
                        await asyncio.sleep(wait_time)

        return np.array(all_embeddings, dtype=np.float32)

    async def get_completion(
        self,
        question: str,
        context: str,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 5000,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Получает ответ от языковой модели.

        Args:
            question: Вопрос пользователя
            context: Контекст для ответа
            model: Модель для генерации
            temperature: Температура генерации (разнообразие)
            max_tokens: Максимальное количество токенов в ответе
            system_prompt: Системный промпт (если None, используется стандартный)

        Returns:
            Сгенерированный ответ
        """
        if system_prompt is None:
            system_prompt = self.default_system_prompt

        user_prompt = f"""
КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ВОПРОС КЛИЕНТА: {question}

Пожалуйста, дай максимально полный и полезный ответ на основе доступной информации."""

        try:
            logger.info("Отправка запроса к LLM...")
            self.stats["completion_calls"] += 1

            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content
        except Exception as e:
            error_message = f"Ошибка при запросе к LLM: {e}"
            logger.error(error_message)
            return error_message

    async def stream_completion(
        self,
        question: str,
        context: str,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> AsyncGenerator[str, None]:
        """
        Стримит ответ от языковой модели по мере генерации.

        Args:
            question: Вопрос пользователя
            context: Контекст для ответа
            model: Модель для генерации
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов в ответе
        """
        system_prompt = (
            self.default_system_prompt
            + """
    ВАЖНОЕ ТРЕБОВАНИЕ ПО ЦИТИРОВАНИЮ:
    Всегда цитируй источники, используя следующий формат:
    1. После каждого утверждения, факта или данных из источника, добавляй номер источника в квадратных скобках [N]
    2. N должно быть номером источника из предоставленного списка (начиная с 1)
    3. Используй только те источники, которые указаны в списке
    4. НЕ добавляй URL в текст, используй только номера в формате [N]
    5. Добавляй ссылку СРАЗУ после текста, к которому она относится
    7. Нумерация источников должна соответствовать нумерации в предоставленном списке
    """
        )

        user_prompt = f"""
    КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
    {context}

    ВОПРОС КЛИЕНТА: {question}

    Пожалуйста, дай максимально полный и полезный ответ на основе доступной информации.
    """

        try:
            logger.info("Отправка запроса к LLM со стримингом...")
            self.stats["completion_calls"] += 1

            stream = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content

        except Exception as e:
            error_message = f"Ошибка при запросе к LLM: {e}"
            logger.error(error_message)
            yield error_message

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику использования API.

        Returns:
            Словарь со статистикой
        """
        return self.stats
