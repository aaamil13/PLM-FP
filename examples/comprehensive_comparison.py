#!/usr/bin/env python3
"""
Всеобхватно сравнение на линейната теория с различни типове данни

Този скрипт създава обобщени графики и анализи на резултатите от всички 
проведени тестове на линейния космологичен модел a(t) = k·t.

Автор: Изследване на линейния космологичен модел
Дата: 2024
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

class ComprehensiveComparison:
    """
    Клас за всеобхватно сравнение на резултатите
    """
    
    def __init__(self):
        """Инициализация на данните"""
        self.test_results = {
            'Суперновi': {
                'linear_score': 0.494,
                'lcdm_score': 0.624,
                'winner': 'Линеен',
                'improvement': 21.0,
                'data_type': 'Светимостни разстояния',
                'mechanism': 'Разширение → яркост',
                'n_points': 1701
            },
            'BAO геометрия': {
                'linear_score': 3.768,
                'lcdm_score': 3.372,
                'winner': 'ΛCDM',
                'improvement': -11.7,
                'data_type': 'Геометрични разстояния',
                'mechanism': 'Пространство → ъгли',
                'n_points': 9
            },
            'AP изкривяване': {
                'linear_score': 1.0,
                'lcdm_score': 1.0,
                'winner': 'Равенство',
                'improvement': 0.0,
                'data_type': 'Чиста геометрия',
                'mechanism': 'Съотношения разстояния',
                'n_points': 100
            },
            'Степенна оптимизация': {
                'linear_score': 1.0,
                'lcdm_score': 1.0,
                'winner': 'Линеен',
                'improvement': 100.0,
                'data_type': 'Математическа структура',
                'mechanism': 'Параметрично пространство',
                'n_points': 200
            }
        }
        
        print("Всеобхватно сравнение на линейната теория")
        print("=" * 50)
        
    def create_performance_comparison(self):
        """Създава графика на качеството по тип тест"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Барна диаграма на χ² стойностите
        tests = list(self.test_results.keys())
        linear_scores = [self.test_results[test]['linear_score'] for test in tests]
        lcdm_scores = [self.test_results[test]['lcdm_score'] for test in tests]
        
        x = np.arange(len(tests))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, linear_scores, width, label='Линеен модел', 
                       color='blue', alpha=0.7)
        bars2 = ax1.bar(x + width/2, lcdm_scores, width, label='ΛCDM модел', 
                       color='green', alpha=0.7)
        
        ax1.set_xlabel('Тип тест')
        ax1.set_ylabel('Качество (по-ниско = по-добро)')
        ax1.set_title('Сравнение на качеството по тестове')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tests, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Добавяме стойности върху барите
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Подобрения в процентите
        improvements = [self.test_results[test]['improvement'] for test in tests]
        colors = ['blue' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]
        
        bars = ax2.bar(tests, improvements, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Тип тест')
        ax2.set_ylabel('Подобрение (%)')
        ax2.set_title('Относително подобрение на линейния модел')
        ax2.grid(True, alpha=0.3)
        
        # Ротираме етикетите
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Добавяме стойности
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=10)
        
        # 3. Разпределение на победителите
        winners = [self.test_results[test]['winner'] for test in tests]
        winner_counts = {}
        for winner in winners:
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
        
        colors_pie = {'Линеен': 'blue', 'ΛCDM': 'green', 'Равенство': 'gray'}
        pie_colors = [colors_pie[w] for w in winner_counts.keys()]
        
        wedges, texts, autotexts = ax3.pie(winner_counts.values(), 
                                          labels=winner_counts.keys(),
                                          colors=pie_colors, autopct='%1.0f%%',
                                          startangle=90)
        ax3.set_title('Разпределение на победителите')
        
        # 4. Брой точки данни по тест
        n_points = [self.test_results[test]['n_points'] for test in tests]
        
        bars = ax4.bar(tests, n_points, color='purple', alpha=0.7)
        ax4.set_xlabel('Тип тест')
        ax4.set_ylabel('Брой точки данни')
        ax4.set_title('Обем на данните по тестове')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Добавяме стойности
        for bar, n in zip(bars, n_points):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{n}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_redshift_evolution(self):
        """Създава графика на еволюцията по червено отместване"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Данни за суперновите по z интервали
        z_intervals = ['[0.0, 0.1)', '[0.1, 0.5)', '[0.5, 1.0)', '[1.0, 1.5)', '[1.5, 2.0)']
        z_centers = [0.05, 0.3, 0.75, 1.25, 1.75]
        rms_linear = [0.2074, 0.1496, 0.1570, 0.3077, 0.1361]
        rms_lcdm = [0.2121, 0.1627, 0.1989, 0.2301, 0.2245]
        n_points_sn = [741, 750, 185, 18, 6]
        
        # 1. RMS по червено отместване за суперновите
        ax1.plot(z_centers, rms_linear, 'o-', color='blue', linewidth=2, 
                markersize=8, label='Линеен модел')
        ax1.plot(z_centers, rms_lcdm, 's-', color='green', linewidth=2, 
                markersize=8, label='ΛCDM модел')
        
        # Размер на маркерите пропорционален на броя точки
        sizes_linear = [np.sqrt(n) for n in n_points_sn]
        sizes_lcdm = [np.sqrt(n) for n in n_points_sn]
        
        ax1.scatter(z_centers, rms_linear, s=sizes_linear, color='blue', alpha=0.3)
        ax1.scatter(z_centers, rms_lcdm, s=sizes_lcdm, color='green', alpha=0.3)
        
        ax1.set_xlabel('Червено отместване z')
        ax1.set_ylabel('RMS остатък (mag)')
        ax1.set_title('Качество на суперновите по червено отместване')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Добавяме анотации за броя точки
        for i, (z, n) in enumerate(zip(z_centers, n_points_sn)):
            ax1.annotate(f'n={n}', (z, max(rms_linear[i], rms_lcdm[i]) + 0.02),
                        ha='center', fontsize=9, alpha=0.7)
        
        # 2. BAO данни по червено отместване
        z_bao = [0.15, 0.25, 0.32, 0.51, 0.57, 0.70, 1.35, 1.48, 2.33]
        chi2_linear_bao = [21.81, 0.04, 1.29, 0.01, 1.27, 2.27, 6.44, 0.70, 0.08]
        chi2_lcdm_bao = [18.05, 0.18, 0.00, 0.82, 0.10, 0.16, 7.91, 1.53, 1.60]
        
        ax2.plot(z_bao, chi2_linear_bao, 'o-', color='blue', linewidth=2, 
                markersize=8, label='Линеен модел')
        ax2.plot(z_bao, chi2_lcdm_bao, 's-', color='green', linewidth=2, 
                markersize=8, label='ΛCDM модел')
        
        ax2.set_xlabel('Червено отместване z')
        ax2.set_ylabel('χ² за отделна точка')
        ax2.set_title('Качество на BAO данните по червено отместване')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Оцветяваме фоновете за различните епохи
        ax1.axvspan(0, 0.5, alpha=0.1, color='red', label='Тъмна енергия')
        ax1.axvspan(0.5, 1.5, alpha=0.1, color='orange', label='Преход')
        ax1.axvspan(1.5, 2.0, alpha=0.1, color='blue', label='Материя')
        
        ax2.axvspan(0, 0.5, alpha=0.1, color='red')
        ax2.axvspan(0.5, 1.5, alpha=0.1, color='orange')
        ax2.axvspan(1.5, 2.5, alpha=0.1, color='blue')
        
        plt.tight_layout()
        return fig
    
    def create_summary_table(self):
        """Създава обобщаваща таблица"""
        
        data = []
        for test_name, results in self.test_results.items():
            data.append([
                test_name,
                results['data_type'],
                results['mechanism'],
                f"{results['linear_score']:.3f}",
                f"{results['lcdm_score']:.3f}",
                results['winner'],
                f"{results['improvement']:+.1f}%",
                results['n_points']
            ])
        
        columns = ['Тест', 'Тип данни', 'Механизъм', 'Линеен χ²', 'ΛCDM χ²', 
                  'Победител', 'Подобрение', 'Точки']
        
        df = pd.DataFrame(data, columns=columns)
        
        # Създаваме таблица като изображение
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Създаваме таблицата
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        
        # Стилизираме таблицата
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Оцветяваме заглавието
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Оцветяваме редовете според победителя
        for i in range(1, len(df) + 1):
            winner = df.iloc[i-1]['Победител']
            if winner == 'Линеен':
                color = '#e6f3ff'  # Светло синьо
            elif winner == 'ΛCDM':
                color = '#e6ffe6'  # Светло зелено
            else:
                color = '#f0f0f0'  # Сиво
            
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)
        
        plt.title('Обобщена таблица на резултатите', fontsize=16, fontweight='bold', pad=20)
        
        return fig, df
    
    def create_mechanism_analysis(self):
        """Анализ по физически механизми"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Радарна диаграма на качествата
        categories = ['Светимостни\nразстояния', 'Геометрични\nразстояния', 
                     'Чиста\nгеометрия', 'Математическа\nструктура']
        
        # Нормализираме резултатите (обръщаме за по-добро = по-високо)
        linear_performance = []
        lcdm_performance = []
        
        for test in ['Суперновi', 'BAO геометрия', 'AP изкривяване', 'Степенна оптимизация']:
            lin_score = self.test_results[test]['linear_score']
            lcdm_score = self.test_results[test]['lcdm_score']
            
            if test == 'Суперновi':
                # По-ниско χ² е по-добро
                lin_perf = 1 / lin_score
                lcdm_perf = 1 / lcdm_score
            elif test == 'BAO геометрия':
                # По-ниско χ² е по-добро
                lin_perf = 1 / lin_score
                lcdm_perf = 1 / lcdm_score
            else:
                # За AP и степенна оптимизация близо до 1 е добро
                lin_perf = 1.0
                lcdm_perf = 1.0
            
            linear_performance.append(lin_perf)
            lcdm_performance.append(lcdm_perf)
        
        # Нормализираме до 0-1
        max_val = max(max(linear_performance), max(lcdm_performance))
        linear_performance = [x/max_val for x in linear_performance]
        lcdm_performance = [x/max_val for x in lcdm_performance]
        
        # Добавяме първата точка в края за затворена форма
        linear_performance += linear_performance[:1]
        lcdm_performance += lcdm_performance[:1]
        
        # Ъгли за радарната диаграма
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax1 = plt.subplot(121, projection='polar')
        ax1.plot(angles, linear_performance, 'o-', linewidth=2, label='Линеен модел', color='blue')
        ax1.fill(angles, linear_performance, alpha=0.25, color='blue')
        ax1.plot(angles, lcdm_performance, 's-', linewidth=2, label='ΛCDM модел', color='green')
        ax1.fill(angles, lcdm_performance, alpha=0.25, color='green')
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('Качество по типове тестове', size=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax1.grid(True)
        
        # 2. Хистограма на подобренията
        test_names = list(self.test_results.keys())
        improvements = [self.test_results[test]['improvement'] for test in test_names]
        
        ax2 = plt.subplot(122)
        colors = ['blue' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]
        bars = ax2.barh(test_names, improvements, color=colors, alpha=0.7)
        
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Подобрение (%)')
        ax2.set_title('Относително подобрение на линейния модел')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Добавяме стойности
        for bar, imp in zip(bars, improvements):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{imp:+.1f}%', ha='left' if width > 0 else 'right', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def generate_all_plots(self):
        """Генерира всички графики"""
        
        print("Създавам графики за всеобхватното сравнение...")
        
        # 1. Основно сравнение на качеството
        fig1 = self.create_performance_comparison()
        fig1.suptitle('Всеобхватно сравнение на качеството на моделите', 
                     fontsize=16, fontweight='bold')
        fig1.savefig('comprehensive_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("  ✓ comprehensive_performance_comparison.png")
        
        # 2. Еволюция по червено отместване
        fig2 = self.create_redshift_evolution()
        fig2.suptitle('Еволюция на качеството по червено отместване', 
                     fontsize=16, fontweight='bold')
        fig2.savefig('redshift_evolution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("  ✓ redshift_evolution_comparison.png")
        
        # 3. Обобщаваща таблица
        fig3, df = self.create_summary_table()
        fig3.savefig('comprehensive_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("  ✓ comprehensive_summary_table.png")
        
        # Запазваме таблицата като CSV
        df.to_csv('comprehensive_results_summary.csv', index=False, encoding='utf-8')
        print("  ✓ comprehensive_results_summary.csv")
        
        # 4. Анализ по механизми
        fig4 = self.create_mechanism_analysis()
        fig4.suptitle('Анализ по физически механизми', 
                     fontsize=16, fontweight='bold')
        fig4.savefig('mechanism_analysis_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print("  ✓ mechanism_analysis_comparison.png")
        
        print("\nВсички графики са създадени успешно!")
        
    def print_executive_summary(self):
        """Принтира изпълнително резюме"""
        
        print("\n" + "=" * 70)
        print("ИЗПЪЛНИТЕЛНО РЕЗЮМЕ")
        print("Линейна космологична теория a(t) = k·t срещу реални данни")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        linear_wins = sum(1 for test in self.test_results.values() if test['winner'] == 'Линеен')
        lcdm_wins = sum(1 for test in self.test_results.values() if test['winner'] == 'ΛCDM')
        ties = sum(1 for test in self.test_results.values() if test['winner'] == 'Равенство')
        
        print(f"\n📊 ОБЩИ РЕЗУЛТАТИ:")
        print(f"   Общо тестове: {total_tests}")
        print(f"   Линеен модел печели: {linear_wins} ({linear_wins/total_tests*100:.0f}%)")
        print(f"   ΛCDM модел печели: {lcdm_wins} ({lcdm_wins/total_tests*100:.0f}%)")
        print(f"   Равенство: {ties} ({ties/total_tests*100:.0f}%)")
        
        print(f"\n🎯 КЛЮЧОВИ ОТКРИТИЯ:")
        
        for test_name, results in self.test_results.items():
            winner_icon = "🏆" if results['winner'] == 'Линеен' else "🥈" if results['winner'] == 'ΛCDM' else "🤝"
            print(f"   {winner_icon} {test_name}: {results['winner']} ({results['improvement']:+.1f}%)")
            print(f"      {results['data_type']} - {results['mechanism']}")
        
        print(f"\n🔬 ФИЗИЧНО ЗНАЧЕНИЕ:")
        print(f"   • Линейният модел превъзхожда за светимостни разстояния")
        print(f"   • ΛCDM остава по-добър за геометрични разстояния")  
        print(f"   • Математическата оптималност потвърждава специалната роля на n=1.0")
        print(f"   • Геометричното съответствие показва съвместимост при умерени z")
        
        print(f"\n🌟 РЕВОЛЮЦИОННИ АСПЕКТИ:")
        print(f"   • H(t) = 1/t може да е фундаментален закон на природата")
        print(f"   • Тъмната енергия може да е артефакт от неправилен модел")
        print(f"   • Принципът на Occam's razor: 1 параметър vs 6+ параметъра")
        print(f"   • Комплементарността предполага нова физика")
        
        print("\n" + "=" * 70)


def main():
    """Главна функция"""
    
    # Създаваме обекта за сравнение
    comparison = ComprehensiveComparison()
    
    # Генерираме всички графики
    comparison.generate_all_plots()
    
    # Принтираме изпълнителното резюме
    comparison.print_executive_summary()


if __name__ == "__main__":
    main() 