import logging
import sys

# Создаём свой логгер с уникальным именем
logger = logging.getLogger("eora")
logger.setLevel(logging.ERROR)  # ловим ERROR и выше

# Проверяем, есть ли уже обработчики, чтобы не добавлять дубли
if not logger.handlers:
    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)

    # Формат сообщений
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Добавляем обработчик к логгеру
    logger.addHandler(console_handler)
