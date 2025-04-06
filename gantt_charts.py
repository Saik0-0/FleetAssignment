import copy

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# текущее распределение вс на рейсы
previous_solution = pd.read_csv('csv_files/new_previous_solution.csv', sep=';')


def gantt_chart(result_schedule: pd.DataFrame, dict_of_swapped: dict, spare_aircraft=None):
    schedule = copy.deepcopy(result_schedule)
    schedule['aircraft_id'] = schedule['aircraft_id'].astype(str)

    if spare_aircraft:
        # Создаем список всех aircraft_id, которые должны быть (14-18)
        all_aircraft_ids = range(14, 19)

        # Находим aircraft_id, которых нет в текущем DataFrame
        missing_ids = set(all_aircraft_ids) - set(schedule['aircraft_id'].unique())

        # Если есть отсутствующие aircraft_id, добавляем их как пустые строки
        if missing_ids:
            # Создаем DataFrame с отсутствующими aircraft_id
            missing_df = pd.DataFrame({
                'aircraft_id': list(missing_ids),
                'departure_time': [schedule['departure_time'].iloc[0]] * len(missing_ids),
                'arrival_time': [schedule['departure_time'].iloc[0]] * len(missing_ids),
                'color': ['white'] * len(missing_ids),  # белый цвет (не будет виден на диаграмме)
                'flight_id': [''] * len(missing_ids),
                'previous_solution_id': [-1] * len(missing_ids)  # или другое значение по умолчанию
            })

            # Объединяем с исходным DataFrame
            schedule = pd.concat([schedule, missing_df], ignore_index=True)
    # Добавляем колонку с цветами
    schedule['color'] = schedule['previous_solution_id'].apply(
        lambda x: 'red' if x in dict_of_swapped['was_triggered']
        else 'green' if x in dict_of_swapped['was_health']
        else 'blue')

    if spare_aircraft:
        sorted_ids = list(range(1, 19))
    else:
        sorted_ids = list(range(1, 14))

    # Строим диаграмму Ганта
    gantt = px.timeline(
        schedule,
        x_start='departure_time',
        x_end='arrival_time',
        y='aircraft_id',
        color='color',  # используем колонку с цветами
        title='Flight schedule',
        hover_name='flight_id',
        text='flight_id',
        color_discrete_map={'red': 'red', 'blue': 'blue', 'green': 'green', 'white': 'white'}
    )

    # Настраиваем порядок категорий для оси y
    gantt.update_yaxes(
        categoryorder='array',
        categoryarray=sorted_ids,
        autorange='reversed'  # если нужно отображать в обратном порядке
    )
    gantt.update_traces(textposition='inside')
    gantt.update_layout(
        yaxis_title='aircraft_id',
        xaxis_title='time',
        showlegend=False
    )
    gantt.show()

