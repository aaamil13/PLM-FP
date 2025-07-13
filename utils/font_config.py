import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform

def setup_cyrillic_fonts():
    """
    Настройка на кирилски фонтове за matplotlib
    """
    # Опитай различни кирилски фонтове в зависимост от ОС
    if platform.system() == 'Windows':
        cyrillic_fonts = [
            'Arial Unicode MS',
            'Calibri',
            'Tahoma',
            'Segoe UI',
            'Times New Roman'
        ]
    else:  # Linux/Mac
        cyrillic_fonts = [
            'Liberation Sans',
            'Ubuntu',
            'Noto Sans',
            'DejaVu Sans'
        ]
    
    # Намери първия достъпен кирилски фонт
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    selected_font = None
    for font in cyrillic_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # Ако не намерим специален фонт, използваме DejaVu Sans
    if not selected_font:
        selected_font = 'DejaVu Sans'
    
    # Настройка на matplotlib
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['font.sans-serif'] = [selected_font]
    plt.rcParams['axes.unicode_minus'] = False
    
    # Кодировка настройки
    plt.rcParams['font.size'] = 10
    
    print(f"Избран фонт: {selected_font}")
    return selected_font

def clear_font_cache():
    """
    Изчисти кеша на фонтовете
    """
    try:
        cache_dir = fm.get_cachedir()
        cache_file = os.path.join(cache_dir, 'fontlist-v330.json')
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print("Кеш на фонтовете изчистен")
    except Exception as e:
        print(f"Грешка при изчистване на кеша: {e}")

def test_cyrillic_rendering():
    """
    Тестова функция за кирилско рендериране
    """
    setup_cyrillic_fonts()
    
    import numpy as np
    
    # Тестови данни
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Синусоида')
    plt.xlabel('Време (сек)')
    plt.ylabel('Амплитуда')
    plt.title('Тест на кирилски фонт')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Изчисти кеша и настрой фонтовете
    clear_font_cache()
    setup_cyrillic_fonts()
    
    # Тестово рендериране
    test_cyrillic_rendering()
