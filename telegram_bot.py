import logging
import re

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger("eora")


class TelegramBot:
    """Класс для управления Telegram-ботом."""

    def __init__(
        self,
        token: str,
        search_engine,
        context_builder,
        ai_client,
    ):
        """
        Инициализация бота.

        Args:
            token: Токен Telegram-бота
            search_engine: Экземпляр SearchEngine
            context_builder: Экземпляр ContextBuilder
            ai_client: Экземпляр AIClient
        """
        self.token = token
        self.search_engine = search_engine
        self.context_builder = context_builder
        self.ai_client = ai_client
        self.application = None

        # Подготавливаем обработчики команд
        self.handlers = {"start": self.start, "help": self.help_command}

    async def start(self, update: Update, context) -> None:
        """Обрабатывает команду /start."""
        await update.message.reply_text(
            "👋 Здравствуйте! Я AI-ассистент компании Eora.\n\n"
        )

    async def help_command(self, update: Update, context) -> None:
        """Обрабатывает команду /help."""
        await update.message.reply_text(
            "🔍 Я могу ответить на вопросы о проектах и услугах компании Eora.\n\n"
        )

    async def handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Обрабатывает входящие сообщения."""
        user_message = update.message.text
        chat_id = update.effective_chat.id

        # Отправляем уведомление о том, что бот печатает
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        try:
            # Поиск релевантной информации
            logger.info(f"Поиск информации по запросу: {user_message}")
            search_results = await self.search_engine.search(user_message, top_k=8)

            if not search_results:
                await update.message.reply_text(
                    "❌ Не удалось найти релевантную информацию по вашему запросу.\n\n"
                    "Попробуйте переформулировать вопрос или использовать другие ключевые слова."
                )
                return

            # Создание контекста для LLM
            context_for_llm = self.context_builder.create_context(
                search_results, user_message
            )

            if not context_for_llm.strip():
                await update.message.reply_text(
                    "❌ Найденная информация недостаточно релевантна для ответа."
                )
                return

            # Получаем источники
            sources = self.context_builder.extract_sources(search_results)

            # Модифицируем контекст, добавляя информацию об источниках и инструкции по форматированию
            sources_info = "\n\nИСТОЧНИКИ ДЛЯ ЦИТИРОВАНИЯ:\n"
            for i, source in enumerate(sources, 1):
                sources_info += f"{i}. {source['name']} - {source['url']}\n"

            formatting_instructions = """
ВАЖНО: При ответе ОБЯЗАТЕЛЬНО соблюдай эти правила для цитирования:
1. Когда ссылаешься на информацию из источников, указывай источник в формате [N]
2. N должно быть номером источника из списка выше (начиная с 1)
3. Добавляй ссылку [N] СРАЗУ после текста, к которому она относится
4. Цитируй каждый источник хотя бы один раз
5. Используй все релевантные источники из предоставленного списка
6. НЕ добавляй URL внутри текста, используй только номера в квадратных скобках
7. НЕ добавляй отдельный список источников в конце ответа
    """

            # Добавляем источники и инструкции к контексту
            enhanced_context = context_for_llm + sources_info + formatting_instructions

            # Отправляем сообщение о генерации ответа
            progress_message = await update.message.reply_text("...")

            # Получаем ответ от LLM со стримингом
            full_response = ""
            message_update_counter = 0

            async for response_chunk in self.ai_client.stream_completion(
                user_message, enhanced_context
            ):
                full_response += response_chunk
                message_update_counter += 1

                # Обновляем сообщение каждые 50 чанков или если чанк содержит знак конца абзаца
                if message_update_counter >= 50 or "\n" in response_chunk:
                    try:
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=progress_message.message_id,
                            text=full_response,
                            disable_web_page_preview=True,
                        )
                        message_update_counter = 0
                    except Exception as e:
                        logger.warning(f"Не удалось обновить сообщение: {e}")

            # После получения полного ответа, обрабатываем ссылки
            # Находим все упоминания [N] в тексте по порядку их появления
            citations = re.findall(r"\[(\d+)\]", full_response)

            # Получаем уникальные номера в порядке их появления в тексте
            unique_citations = []
            for citation in citations:
                if citation not in unique_citations:
                    unique_citations.append(citation)

            # Создаем маппинг старых номеров на новые последовательные номера
            citation_mapping = {old: i + 1 for i, old in enumerate(unique_citations)}

            # Создаем маппинг новых номеров на URL источников
            url_mapping = {}
            for old_num, new_num in citation_mapping.items():
                old_num_int = int(old_num)
                if 1 <= old_num_int <= len(sources):
                    url = sources[old_num_int - 1]["url"]
                    url_mapping[new_num] = url

            # Функция для замены ссылок на последовательные номера
            def replace_citation(match):
                old_num = match.group(1)
                if old_num in citation_mapping:
                    new_num = citation_mapping[old_num]
                    if new_num in url_mapping:
                        return f"\[[{new_num}]({url_mapping[new_num]})]"
                    else:
                        return f"[{new_num}]"
                return match.group(0)

            # Заменяем все ссылки [N] на последовательные номера с URL
            processed_response = re.sub(r"\[(\d+)\]", replace_citation, full_response)

            # Отправляем финальное сообщение с обработанными ссылками
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=progress_message.message_id,
                text=processed_response,
                parse_mode="markdown",
                disable_web_page_preview=True,
            )

        except Exception as e:
            logger.error(f"Произошла ошибка при обработке сообщения: {e}")
            await update.message.reply_text(
                f"❌ Произошла ошибка при обработке вашего запроса: {str(e)}\n\n"
                "Пожалуйста, попробуйте еще раз позже или обратитесь к администратору."
            )

    async def error_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Обрабатывает ошибки в боте."""
        logger.error(f"Ошибка при обработке обновления {update}: {context.error}")

        try:
            # Отправляем сообщение об ошибке пользователю
            if update and update.effective_chat:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
                )
        except Exception as e:
            logger.error(f"Не удалось отправить сообщение об ошибке: {e}")

    async def setup(self):
        """Настраивает и запускает бота."""
        # Создаем и настраиваем бота
        self.application = Application.builder().token(self.token).build()

        # Регистрируем обработчики команд
        for command, handler in self.handlers.items():
            self.application.add_handler(CommandHandler(command, handler))

        # Регистрируем обработчик сообщений
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        # Регистрируем обработчик ошибок
        self.application.add_error_handler(self.error_handler)

        # Инициализируем бота
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

        logger.info("Бот успешно запущен и ожидает сообщений!")
        print("✅ Бот успешно запущен и ожидает сообщений!")
        print("Для остановки нажмите Ctrl+C")

    async def shutdown(self):
        """Корректно останавливает бота."""
        if self.application:
            logger.info("Остановка бота...")
            await self.application.updater.stop_polling()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Бот остановлен.")
