import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# текущее распределение вс на рейсы
previous_solution = pd.read_csv('csv_files/new_previous_solution.csv', sep=';')


def gantt_chart(schedule: pd.DataFrame, dict_of_swapped: dict):
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
    # Добавляем колонку с цветами: если previous_solution_id == 1, то красный, иначе синий
    schedule['color'] = schedule['previous_solution_id'].apply(
        lambda x: 'red' if x in dict_of_swapped['was_triggered']
        else 'green' if x in dict_of_swapped['was_health']
        else 'blue')

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

    gantt.update_traces(textposition='inside')
    gantt.update_layout(yaxis_title='aircraft_id', xaxis_title='time', showlegend=False)
    gantt.update_yaxes(autorange="reversed")
    gantt.show()


#
# fig, ax = plt.subplots(figsize=(30, 10))
#
# for i in range(len(previous_solution)):
#     ax.barh(previous_solution['aircraft_id'][i],
#             arrival_time[i] - departure_time[i],
#             left=departure_time[i],
#             color='skyblue',
#             edgecolor='black')
#
# for i in range(len(previous_solution)):
#     ax.text(departure_time[i] + (arrival_time[i] - departure_time[i]) / 2,
#             i,
#             previous_solution['flight_id'][i],
#             ha='center',
#             va='center',
#             color='black')
#
# ax.set_xlabel('time')
# ax.set_ylabel('aircraft_id')
# ax.set_title('Gantt Chart')
#
# plt.show()