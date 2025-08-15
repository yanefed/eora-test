import asyncio
import re
from typing import Dict, List, Optional

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

from logger import logger


class WebScraper:
    """Класс для загрузки и парсинга веб-страниц."""

    def __init__(self, headers: Dict[str, str]):
        """
        Инициализация WebScraper.

        Args:
            headers: Заголовки для HTTP запросов
        """
        self.headers = headers

    async def scrape_url(
        self, client: httpx.AsyncClient, url: str
    ) -> Optional[Dict[str, str]]:
        """
        Асинхронно загружает и извлекает основной текст со страницы.

        Args:
            client: Асинхронный HTTP клиент
            url: URL для парсинга

        Returns:
            Словарь с содержимым страницы и источником, или None в случае ошибки
        """
        try:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(
                response.content, "lxml"
            )  # Используем lxml для скорости
            main_content = soup.find("main") or soup.find("body")

            if main_content:
                # Удаляем ненужные элементы
                for tag in main_content(
                    ["script", "style", "nav", "footer", "header", "aside"]
                ):
                    tag.decompose()

                # Извлекаем текст и очищаем его
                text = main_content.get_text(separator=" ", strip=True)
                clean_text = re.sub(r"\s+", " ", text)

                return {"page_content": clean_text, "source": url}

            return None
        except httpx.RequestError as e:
            logger.error(f"Ошибка при парсинге URL {url}: {e}")
            return None

    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Асинхронно парсит список URL.

        Args:
            urls: Список URL для парсинга

        Returns:
            Список словарей с содержимым страниц
        """
        scraped_docs = []

        async with httpx.AsyncClient(
            headers=self.headers, timeout=30, follow_redirects=True
        ) as client:
            tasks = [self.scrape_url(client, url) for url in urls]

            for future in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Парсинг URL",
            ):
                result = await future
                if result:
                    scraped_docs.append(result)

        return scraped_docs
