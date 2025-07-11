# Линеен космологичен модел - Тест на времеви пространствен модел без тъмна енергия

## Описание

Този проект изследва алтернативен космологичен модел, базиран на линейно разширение на Вселената без тъмна енергия. Моделът предполага, че мащабният фактор a(t) е пропорционален на времето t, което води до интересни следствия за еволюцията на времето и плътността.

## Основни хипотези

1. **Линейно разширение**: a(t) = k × t
2. **Кубично ускорение на времето**: τ(t) ∝ 1/t³
3. **Отсъствие на тъмна енергия**: Моделът използва само обикновена материя

## Структура на проекта

```
Test_5/
├── docs/                          # Документация
│   ├── theory/                    # Теоретични основи
│   ├── mathematical_model/        # Математически модел
│   └── comparisons/              # Сравнения със стандартни модели
├── analysis/                     # Анализни модули
│   ├── cosmological_model/       # Космологичен анализ
│   ├── time_evolution/           # Времева еволюция
│   └── density_analysis/         # Анализ на плътността
├── lib/                          # Основни библиотеки
│   ├── core/                     # Основни функции
│   ├── utils/                    # Помощни функции
│   └── visualization/            # Визуализация
├── verification/                 # Верификационни тестове
│   ├── model_tests/              # Тестове на модела
│   ├── boundary_tests/           # Гранични тестове
│   └── consistency_tests/        # Тестове за съгласуваност
├── tests/                        # Тестове
│   ├── unit/                     # Unit тестове
│   ├── integration/              # Интеграционни тестове
│   └── performance/              # Performance тестове
├── examples/                     # Примери
├── scripts/                      # Скриптове
└── Base.md                       # Базово разсъждение
```

## Инсталация

1. Клонирайте репозиторието:
```bash
git clone [repository-url]
cd Test_5
```

2. Създайте виртуална среда:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

3. Инсталирайте зависимостите:
```bash
pip install -r requirements.txt
```

## Бързо започване

### Основни изчисления

```python
from lib.core import linear_scale_factor, hubble_parameter, matter_density

# Времеви масив
t = np.linspace(0.1, 10, 100)

# Мащабен фактор
a = linear_scale_factor(t, k=1.0)

# Параметър на Хъбъл
H = hubble_parameter(t)

# Плътност на материята
rho = matter_density(t, rho0=1.0)
```

### Анализ на времевата еволюция

```python
from analysis.time_evolution.temporal_dynamics import TemporalAnalyzer, TemporalParameters

# Създаване на параметри
params = TemporalParameters(tau0=1.0, rho0=1.0, a0=1.0, k=1.0)

# Анализатор
analyzer = TemporalAnalyzer(params)

# Темпо на времето
tempo = analyzer.time_tempo(t)

# Графики
fig, axes = analyzer.plot_temporal_evolution(t_max=10.0)
```

### Анализ на плътността

```python
from analysis.density_analysis.density_evolution import DensityAnalyzer, DensityParameters

# Параметри
params = DensityParameters(rho0=1.0, a0=1.0, k=1.0)

# Анализатор
analyzer = DensityAnalyzer(params)

# Еволюция на плътността
fig, axes = analyzer.plot_density_evolution(t_max=10.0)
```

## Ключови резултати

### Математически модел

- **Мащабен фактор**: a(t) = k × t
- **Плътност**: ρ(t) = ρ₀ × (a₀/(k×t))³
- **Темпо на времето**: τ(t) = τ₀ × (a₀/(k×t))³
- **Параметър на Хъбъл**: H(t) = 1/t

### Физическа интерпретация

1. **Времето се ускорява**: В ранната Вселена времето е текло бавно
2. **Плътността намалява кубично**: По-бързо от стандартните модели
3. **Постоянна скорост на разширение**: Без забавяне или ускорение

## Тестване

Изпълнете тестовете:

```bash
# Всички тестове
pytest tests/

# Unit тестове
pytest tests/unit/

# Верификационни тестове
pytest verification/

# С покритие
pytest --cov=lib tests/
```

## Документация

Подробната документация е налична в директория `docs/`:

- [Теоретични основи](docs/theory/linear_expansion_hypothesis.md)
- [Математически модел](docs/mathematical_model/equations.md)
- [Сравнения със стандартни модели](docs/comparisons/standard_model_vs_linear.md)

## Примери

Проверете директория `examples/` за подробни примери на употреба.

## Допринасяне

1. Fork проекта
2. Създайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit промените (`git commit -m 'Add some AmazingFeature'`)
4. Push към branch (`git push origin feature/AmazingFeature`)
5. Отворете Pull Request

## Лиценз

Този проект е с отворен код. Вижте LICENSE файла за подробности.

## Контакти

За въпроси и предложения: [email]

## Цитиране

Ако използвате този код в научна работа, моля цитирайте:

```
@misc{linear_cosmology_model,
  title={Linear Cosmological Model: A Test of Temporal-Spatial Model Without Dark Energy},
  author={[Authors]},
  year={2025},
  url={[repository-url]}
}
``` 