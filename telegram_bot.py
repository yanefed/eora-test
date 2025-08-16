import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
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

            # Отправляем сообщение о генерации ответа
            progress_message = await update.message.reply_text("...")

            # Получаем ответ от LLM со стримингом
            full_response = ""
            message_update_counter = 0
            async for response_chunk in self.ai_client.stream_completion(
                user_message, context_for_llm
            ):
                full_response += response_chunk
                message_update_counter += 1

                # Обновляем сообщение каждые 10 чанков или если чанк содержит знак конца абзаца
                if message_update_counter >= 10 or "\n" in response_chunk:
                    try:
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=progress_message.message_id,
                            text=full_response,
                            parse_mode="markdown",
                        )
                        message_update_counter = 0
                    except Exception as e:
                        logger.warning(f"Не удалось обновить сообщение: {e}")

            # Обновляем сообщение с полным ответом в конце
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=progress_message.message_id,
                    text=full_response,
                    parse_mode="markdown",
                )
            except Exception as e:
                logger.warning(f"Не удалось обновить финальное сообщение: {e}")

            # Создаем и отправляем клавиатуру с источниками
            sources = self.context_builder.extract_sources(search_results)
            keyboard = [
                [InlineKeyboardButton(s["name"], url=s["url"])] for s in sources
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                "📚 Источники информации:", reply_markup=reply_markup
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
