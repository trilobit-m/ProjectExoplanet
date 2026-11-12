"""
BLS Pipeline для поиска транзитных экзопланет в данных TESS
Минимальная версия — только анализ и вывод результатов в консоль.
"""

import lightkurve as lk
import numpy as np
import pandas as pd
import astropy.units as u
import warnings

warnings.filterwarnings('ignore')


def analyze_star(tic_id, min_period=0.5, max_period=10.0, 
                 sde_threshold=10.0, depth_threshold=0.05):
    """
    Анализирует кривую блеска одной звезды методом BLS.
    
    Parameters
    ----------
    tic_id : int
        Номер звезды в каталоге TESS Input Catalog (TIC)
    min_period : float
        Минимальный искомый период (дни)
    max_period : float
        Максимальный искомый период (дни)
    sde_threshold : float
        Порог SDE для объявления кандидата (>10 = надёжное обнаружение)
    depth_threshold : float
        Максимальная глубина транзита (0.05 = 5%)
        Значения выше характерны для затменных двойных, не планет
    
    Returns
    -------
    dict : словарь с результатами
    """
    try:
        # Шаг 1: Поиск данных в архиве TESS
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        
        if len(search_result) == 0:
            return {'tic': tic_id, 'status': 'NO_DATA'}
        
        # Шаг 2: Скачивание кривой блеска
        lc = search_result[0].download()
        
        # Шаг 3: Очистка от трендов
        lc_flat = lc.flatten(window_length=501)
        
        # Шаг 4: Извлечение массивов времени и яркости
        time_days = lc_flat.time.value
        flux = lc_flat.flux.value
        
        # Шаг 5: Создание LightCurve с единицами измерения
        lc_clean = lk.LightCurve(time=time_days * u.day, flux=flux)
        
        # Шаг 6: Запуск BLS
        bls = lc_clean.to_periodogram(
            method='bls',
            minimum_period=min_period,
            maximum_period=max_period,
            frequency_factor=500
        )
        
        # Шаг 7: Извлечение параметров
        sde = float(bls.max_power)
        period = float(bls.period_at_max_power.value)
        transit_time = bls.transit_time_at_max_power.value
        
        # Шаг 8: Фильтр по SDE
        if sde < sde_threshold:
            return {'tic': tic_id, 'status': 'REJECTED_LOW_SDE', 'sde': sde}
        
        # Шаг 9: Сворачивание кривой
        folded = lc_clean.fold(period=period, epoch_time=transit_time)
        
        # Шаг 10: Вычисление глубины транзита
        mask = (folded.phase >= -0.1) & (folded.phase <= 0.1)
        depth = 1.0 - float(np.nanmedian(folded.flux[mask]))
        
        # Шаг 11: Фильтр по глубине (отсев двойных)
        if depth > depth_threshold:
            return {'tic': tic_id, 'status': 'REJECTED_DEEP_TRANSIT', 
                    'sde': sde, 'depth': depth}
        
        # Шаг 12: Кандидат найден!
        return {
            'tic': tic_id,
            'status': 'CANDIDATE',
            'sde': sde,
            'period': period,
            'depth': depth
        }
        
    except Exception as e:
        return {'tic': tic_id, 'status': 'ERROR', 'error_msg': str(e)}


def run_pipeline(csv_file, sde_threshold=10.0, depth_threshold=0.05):
    """
    Запускает BLS-конвейер на списке звёзд из CSV-файла.
    
    Parameters
    ----------
    csv_file : str
        Путь к CSV с колонкой 'TIC_ID'
    sde_threshold : float
        Порог SDE
    depth_threshold : float
        Максимальная глубина транзита
    
    Returns
    -------
    pd.DataFrame : таблица результатов
    """
    # Загрузка списка звёзд
    df_targets = pd.read_csv(csv_file)
    df_targets['TIC_ID'] = df_targets['TIC_ID'].astype(int)
    
    n_stars = len(df_targets)
    print(f" Запуск BLS-конвейера на {n_stars} звёздах...")
    print(f"   Порог SDE: {sde_threshold}")
    print(f"   Макс. глубина: {depth_threshold*100}%")
    print("=" * 70)
    
    results_list = []
    
    # Цикл по всем звёздам
    for idx, row in df_targets.iterrows():
        tic_id = int(row['TIC_ID'])
        
        # Анализ звезды
        result = analyze_star(
            tic_id=tic_id,
            min_period=0.5,
            max_period=10.0,
            sde_threshold=sde_threshold,
            depth_threshold=depth_threshold
        )
        
        results_list.append(result)
        
        # Вывод результата
        status = result['status']
        
        if status == 'CANDIDATE':
            sde = result['sde']
            period = result['period']
            depth = result['depth']
            print(f"[{idx+1}/{n_stars}]  TIC {tic_id}: КАНДИДАТ | SDE={sde:.2f}, P={period:.4f} дн, глубина={depth*100:.3f}%")
        elif status == 'REJECTED_LOW_SDE':
            sde = result['sde']
            print(f"[{idx+1}/{n_stars}]  TIC {tic_id}: Отсеян (SDE={sde:.3f} < {sde_threshold})")
        elif status == 'REJECTED_DEEP_TRANSIT':
            sde = result['sde']
            depth = result['depth']
            print(f"[{idx+1}/{n_stars}]  TIC {tic_id}: Отсеян (глубина={depth*100:.2f}% > {depth_threshold*100}%)")
        elif status == 'NO_DATA':
            print(f"[{idx+1}/{n_stars}]   TIC {tic_id}: Нет данных TESS")
        elif status == 'ERROR':
            print(f"[{idx+1}/{n_stars}]  TIC {tic_id}: Ошибка — {result.get('error_msg', 'неизвестно')}")
    
    # Итоговая статистика
    print("\n" + "=" * 70)
    
    df_results = pd.DataFrame(results_list)
    
    n_total = len(df_results)
    n_candidates = len(df_results[df_results['status'] == 'CANDIDATE'])
    n_rejected_sde = len(df_results[df_results['status'] == 'REJECTED_LOW_SDE'])
    n_rejected_depth = len(df_results[df_results['status'] == 'REJECTED_DEEP_TRANSIT'])
    n_no_data = len(df_results[df_results['status'] == 'NO_DATA'])
    n_errors = len(df_results[df_results['status'] == 'ERROR'])
    
    print(f"\n Статистика:")
    print(f"    Всего звёзд: {n_total}")
    print(f"    Кандидаты: {n_candidates}")
    print(f"    Отсеяно (слабый SDE): {n_rejected_sde}")
    print(f"    Отсеяно (глубокий транзит): {n_rejected_depth}")
    print(f"    Нет данных: {n_no_data}")
    print(f"    Ошибки: {n_errors}")
    
    # Вывод таблицы кандидатов
    candidates = df_results[df_results['status'] == 'CANDIDATE']
    if len(candidates) > 0:
        print(f"\n НАЙДЕНО {len(candidates)} КАНДИДАТОВ В ЭКЗОПЛАНЕТЫ:")
        print(candidates[['tic', 'sde', 'period', 'depth']].to_string(index=False))
    else:
        print(f"\n Кандидатов в экзопланеты не найдено.")
        print("   Все сигналы с высоким SDE оказались затменными двойными")
        print("   или молодыми переменными звёздами и были отсеяны фильтром по глубине.")
    
    return df_results


# Точка входа
if __name__ == '__main__':
    input_csv = "target_stars_with_data.csv"
    
    try:
        results = run_pipeline(
            csv_file=input_csv,
            sde_threshold=10.0,
            depth_threshold=0.05
        )
    except FileNotFoundError:
        print(f" Файл '{input_csv}' не найден!")
        print("   Создайте CSV-файл со списком звёзд (колонка 'TIC_ID')")
