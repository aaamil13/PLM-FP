import sys
import locale
import os

def setup_cp1251_environment():
    """
    Настройка на средата за работа с cp1251
    """
    # Настройка на системната кодировка
    if sys.platform.startswith('win'):
        # Windows специфични настройки
        import codecs
        
        # Регистрирай cp1251 codec ако не е достъпен
        try:
            'тест'.encode('cp1251')
        except LookupError:
            codecs.register(lambda name: codecs.lookup('utf-8') if name == 'cp1251' else None)
    
    # Настройка на locale
    try:
        if sys.platform.startswith('win'):
            locale.setlocale(locale.LC_ALL, 'Bulgarian_Bulgaria.1251')
        else:
            locale.setlocale(locale.LC_ALL, 'bg_BG.UTF-8')
    except locale.Error:
        print("Предупреждение: Не може да се настрои българска локализация")
    
    # Настройка на environment variables
    os.environ['PYTHONIOENCODING'] = 'cp1251'
    
    print("CP1251 среда настроена успешно")

def safe_decode(text, encoding='cp1251'):
    """
    Безопасно декодиране на текст
    """
    if isinstance(text, bytes):
        try:
            return text.decode(encoding)
        except UnicodeDecodeError:
            # Fallback към utf-8
            try:
                return text.decode('utf-8')
            except UnicodeDecodeError:
                # Последен опит с игнориране на грешки
                return text.decode(encoding, errors='ignore')
    return text

def safe_encode(text, encoding='cp1251'):
    """
    Безопасно кодиране на текст
    """
    if isinstance(text, str):
        try:
            return text.encode(encoding)
        except UnicodeEncodeError:
            # Fallback към utf-8
            try:
                return text.encode('utf-8')
            except UnicodeEncodeError:
                # Последен опит с игнориране на грешки
                return text.encode(encoding, errors='ignore')
    return text

def print_cp1251(text):
    """
    Принтиране на cp1251 текст
    """
    try:
        if isinstance(text, str):
            print(text.encode('cp1251').decode('cp1251'))
        else:
            print(safe_decode(text))
    except Exception as e:
        print(f"Грешка при принтиране: {e}")
        print(text)

# Инициализация при импортиране
setup_cp1251_environment()
