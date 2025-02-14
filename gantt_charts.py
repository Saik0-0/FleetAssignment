import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# текущее распределение вс на рейсы
previous_solution = pd.read_csv('csv_files/df_previous_solution.csv', sep=';')

gantt = px.timeline(previous_solution,
                    x_start='departure_time',
                    x_end='arrival_time',
                    y='aircraft_id',
                    title='Flight schedule',
                    hover_name='flight_id',
                    text='flight_id')
gantt.update_traces(textposition='inside', marker=dict(color='Blue'))
gantt.update_layout(yaxis_title='aircraft_id', xaxis_title='time', showlegend=False)
gantt.update_yaxes(autorange="reversed")
gantt.show()

# arrival_time = pd.to_datetime(previous_solution['arrival_time'])
# departure_time = pd.to_datetime(previous_solution['departure_time'])
#
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