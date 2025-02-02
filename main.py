import pandas as pd
from ast import literal_eval
from datetime import datetime
from dateutil import parser

'''Добавить проверку что указанные в equipment_disrupted_flights индексы есть в problematic_aircraft_equipment'''

# парк доступных вс
fleet = pd.read_csv('csv_files/df_aircraft.csv', sep=';')
# авиарейсы
flights = pd.read_csv('csv_files/df_flights.csv', sep=';')
# требования к вс на рейсе
flight_equipments = pd.read_csv('csv_files/df_flight_equipments.csv', sep=';')
# текущее распределение вс на рейсы
previous_solution = pd.read_csv('csv_files/df_previous_solution.csv', sep=';')
# тех обслуживание вс
technical_service = pd.read_csv('csv_files/df_technical_service.csv', sep=';')

# disruption: изменение оснащения вс
problematic_aircraft_equipment = pd.read_csv('csv_files/df_problematic_aircraft_equipment.csv', sep=';')
# disruption: перенос рейсов
problematic_flight_shift = pd.read_csv('csv_files/df_problematic_flight_shift.csv', sep=';')


def equipment_disrupted_flights(flight_equipments_table: pd.DataFrame, previous_solution_table: pd.DataFrame,
                                disruptions_table: pd.DataFrame, *disruption_ids: int) -> list:
    disruptions_table = disruptions_table.loc[[*disruption_ids]]
    problematic_aircrafts = disruptions_table['aircraft_id'].tolist()

    # Структура словаря: ключи - id самолетов с изменением оборудования,
    # значения - индексы нового оборудования
    new_equipment_dict = {}
    for aircraft_id in problematic_aircrafts:
        new_equipment_dict[aircraft_id] = disruptions_table[disruptions_table['aircraft_id'] == aircraft_id]['equipment_id'].tolist()[0]

    problematic_flights = previous_solution_table[previous_solution_table['aircraft_id'].isin(problematic_aircrafts)]

    # Структура словаря: ключи - id самолетов с изменением оборудования,
    # значения - set из id рейсов куда распределены самолеты
    problematic_flights_id = {}
    for aircraft_id in problematic_aircrafts:
        problematic_flights_id[aircraft_id] = set(problematic_flights[problematic_flights['aircraft_id'] == aircraft_id]['flight_id'].tolist())

    disrupted_flights = []
    for aircraft_id in problematic_aircrafts:
        flight_equipments_subtable = flight_equipments_table[flight_equipments_table['flight_id'].isin(problematic_flights_id[aircraft_id])]
        for flight_id in problematic_flights_id[aircraft_id]:
            if new_equipment_dict[aircraft_id] not in literal_eval(flight_equipments_subtable[flight_equipments_subtable['flight_id'] == flight_id]['equipment_ids'].iloc[0]):
                disrupted_flights.append((flight_id, aircraft_id))

    return disrupted_flights


def get_time_from_table(table: pd.DataFrame, previous_solution_id: int, column_name: str) -> datetime:
    if 'new' not in column_name:
        previous_solution_id += 1
    string_time = table[table['previous_solution_id'] == previous_solution_id][column_name].iloc[0]

    time_object = parser.parse(string_time, dayfirst=True)
    return time_object


def flight_shift_disrupted_flights(previous_solution_table: pd.DataFrame, disruptions_table: pd.DataFrame,
                                   *disruption_ids: int) -> list:
    """Надо проверять остается ли окно в 50 мин до след рейса"""
    problematic_aircrafts = disruptions_table['aircraft_id'].tolist()
    disrupted_flights = []

    for disruption_id in disruption_ids:
        previous_solution_id = disruptions_table[disruptions_table['problematic_flight_shift_id'] == disruption_id]['previous_solution_id'].iloc[0]

        new_arrival_time = get_time_from_table(disruptions_table, previous_solution_id, 'new_arrival_time')
        next_departure_time = get_time_from_table(previous_solution_table, previous_solution_id, 'departure_time')

        flight_id = previous_solution_table[previous_solution_table['previous_solution_id'] == previous_solution_id]['flight_id'].iloc[0]
        aircraft_id = problematic_aircrafts[disruption_id]

        time_delta = next_departure_time - new_arrival_time
        # 50 минут = 3000 сек - минимальное окно между рейсами
        if time_delta.total_seconds() < 3000:
            disrupted_flights.append((flight_id, aircraft_id))

    return disrupted_flights
