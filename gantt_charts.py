import copy
import pandas as pd
import plotly.express as px
import matplotlib


# Генератор цветов
def get_color_palette(n):
    cmap = matplotlib.colormaps['hsv'].resampled(n)
    return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]


def gantt_chart(result_schedule: pd.DataFrame, dict_of_swapped: dict, spare_aircraft=None):
    schedule = copy.deepcopy(result_schedule)
    schedule['aircraft_id'] = schedule['aircraft_id'].astype(str)

    # === 1. Подготовка структуры с уникальными цветами для подгрупп ===
    color_map = {}  # ключ: previous_solution_id, значение: цвет
    # Генерируем цвета (по количеству всех подсписков)
    all_groups = dict_of_swapped['was_triggered'] + dict_of_swapped['was_health']
    colors = get_color_palette(len(all_groups))

    for group_idx, group in enumerate(all_groups):
        for previous_solution_id in group:
            color_map[previous_solution_id] = colors[group_idx]

    # === 2. Обработка spare_aircraft ===
    if spare_aircraft:
        all_aircraft_ids = range(14, 19)
        missing_ids = set(all_aircraft_ids) - set(schedule['aircraft_id'].unique())
        if missing_ids:
            base_time = schedule['departure_time'].iloc[0]
            missing_df = pd.DataFrame({
                'aircraft_id': list(missing_ids),
                'departure_time': [base_time] * len(missing_ids),
                'arrival_time': [base_time] * len(missing_ids),
                'color': ['white'] * len(missing_ids),
                'flight_id': [''] * len(missing_ids),
                'previous_solution_id': [-1] * len(missing_ids)
            })
            schedule = pd.concat([schedule, missing_df], ignore_index=True)

    # === 3. Назначаем цвет каждому рейсу по словарю color_map ===
    schedule['color'] = schedule['previous_solution_id'].apply(
        lambda x: color_map.get(int(x), '#ADD8E6')  # если не найден в словаре — синий
    )

    # === 4. Формируем поля для подписей ===
    # Если нет колонки arrival_airport / departure_airport, подставим фиктивные
    if 'arrival_airport_code' not in schedule.columns:
        schedule['arrival_airport_code'] = ''
    if 'departure_airport_code' not in schedule.columns:
        schedule['departure_airport_code'] = ''

    # Переводим время в удобный формат
    schedule['departure_str'] = pd.to_datetime(schedule['departure_time']).dt.strftime('%H:%M')
    schedule['arrival_str'] = pd.to_datetime(schedule['arrival_time']).dt.strftime('%H:%M')

    if spare_aircraft:
        sorted_ids = list(range(1, 19))
    else:
        sorted_ids = list(range(1, 14))
    # Подготовим многострочный текст
    schedule['text_block'] = schedule.apply(
        lambda row: (
                f"{row['departure_airport_code']} {row['departure_str']}   "
                + "<br>" + f"   {row['arrival_str']} {row['arrival_airport_code']} "  # третья строка
        ),
        axis=1
    )

    gantt = px.timeline(
        schedule,
        x_start='departure_time',
        x_end='arrival_time',
        y='aircraft_id',
        color='color',
        hover_name='flight_id',
        text='text_block',  # <-- Многострочный текст
        color_discrete_sequence=schedule['color'].unique()
    )

    gantt.update_traces(
        textposition='inside',
        texttemplate='%{text}',  # Используем наш многострочный текст
        insidetextanchor='middle',
        textfont=dict(size=12, color='black'),
        textangle=0  # <-- вот это фиксирует горизонтальный угол
    )
    # Настраиваем порядок категорий для оси y
    gantt.update_yaxes(
        categoryorder='array',
        categoryarray=sorted_ids,
        autorange='reversed'  # если нужно отображать в обратном порядке
    )

    gantt.update_layout(
        yaxis_title='aircraft_id',
        xaxis_title='time',
        showlegend=False,
        uniformtext_minsize=12,  # тут регулируй размер
        uniformtext_mode='show'
    )
    gantt.show()