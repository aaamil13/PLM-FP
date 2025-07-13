import logging
import sys
from logging.handlers import RotatingFileHandler
import codecs

def setup_cp1251_logger(name='app', level=logging.INFO):
    """
    Настройка на logger за работа с cp1251
    """
    # Създай logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Избегни дублиране на handlers
    if logger.handlers:
        return logger
    
    # Formatter с cp1251 поддръжка
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler с cp1251
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Опитай се да настроиш кодировката на конзолата
    try:
        if sys.platform.startswith('win'):
            # Windows конзола
            console_handler.stream = codecs.getwriter('cp1251')(sys.stdout.buffer)
        else:
            # Unix/Linux конзола
            console_handler.stream = codecs.getwriter('utf-8')(sys.stdout.buffer)
    except Exception as e:
        print(f"Предупреждение: Не може да се настрои кодировката на конзолата: {e}")
    
    # File handler с cp1251
    try:
        file_handler = RotatingFileHandler(
            'app.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='cp1251'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Предупреждение: Не може да се създаде файлов handler: {e}")
    
    logger.addHandler(console_handler)
    
    return logger

def log_safe(logger, level, message):
    """
    Безопасно логване на съобщения
    """
    try:
        # Конвертирай съобщението в string ако не е
        if not isinstance(message, str):
            message = str(message)
        
        # Логвай в зависимост от level
        if level == logging.DEBUG:
            logger.debug(message)
        elif level == logging.INFO:
            logger.info(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
        elif level == logging.CRITICAL:
            logger.critical(message)
        else:
            logger.info(message)
            
    except Exception as e:
        # Fallback към обикновен print
        print(f"Logging error: {e}")
        print(f"Original message: {message}")

# Пример за използване
if __name__ == "__main__":
    # Настройка на logger
    logger = setup_cp1251_logger('test_logger')
    
    # Тестови съобщения
    log_safe(logger, logging.INFO, "Тестово съобщение на кирилица")
    log_safe(logger, logging.DEBUG, "Debug съобщение")
    log_safe(logger, logging.WARNING, "Предупреждение")
    log_safe(logger, logging.ERROR, "Грешка")
