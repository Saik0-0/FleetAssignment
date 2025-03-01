from itertools import chain

import pandas as pd
from ast import literal_eval
from datetime import datetime, timedelta
from dateutil import parser

'''
Добавить проверку что переданные в equipment_disrupted_flights() индексы есть в df_problematic_aircraft_equipment.csv
'''

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


def nearest_flights_selection(previous_solution_table: pd.DataFrame,
                              current_time: datetime,
                              *base_airports: str) -> pd.DataFrame:
    """Возвращает часть расписания (указанная дата + 3 суток)"""
    previous_solution_table_result = pd.DataFrame(columns=previous_solution_table.columns)
    for index, flight_row in previous_solution_table.iterrows():
        if current_time <= pd.to_datetime(flight_row['departure_time'], format='%Y-%m-%d %H:%M:%S') <= current_time + timedelta(days=2, hours=23, minutes=59):
            if pd.to_datetime(previous_solution_table.iloc[index + 1]['departure_time']) > current_time + timedelta(days=2, hours=23, minutes=59) and flight_row['arrival_airport_code'] not in base_airports:
                previous_solution_table_result.loc[len(previous_solution_table_result.index)] = flight_row
                sub_index = index + 1
                base_airport_flag = 1
                if sub_index < len(previous_solution_table):
                    while base_airport_flag:
                        if previous_solution_table.iloc[sub_index]['departure_airport_code'] in base_airports:
                            base_airport_flag = 0
                        else:
                            previous_solution_table_result.loc[len(previous_solution_table_result.index)] = previous_solution_table.iloc[sub_index]
                            if sub_index < len(previous_solution_table) - 1:
                                sub_index += 1
                            else:
                                base_airport_flag = 0
                continue
            if index == 0:
                if flight_row['departure_airport_code'] not in base_airports:
                    continue
            elif not any((previous_solution_table_result == previous_solution_table.iloc[index - 1]).all(axis=1)) and flight_row['departure_airport_code'] not in base_airports:
                continue
            previous_solution_table_result.loc[len(previous_solution_table_result.index)] = flight_row

    return previous_solution_table_result


def equipment_disrupted_flights(flight_equipments_table: pd.DataFrame,
                                previous_solution_table: pd.DataFrame,
                                disruptions_table: pd.DataFrame,
                                current_time: datetime,
                                *disruption_ids: int) -> list:
    """Возвращает список полётов, которые становятся невыполнимыми из-за смены оснащения ВС"""
    disruptions_table = disruptions_table.loc[[*disruption_ids]]
    problematic_aircrafts = disruptions_table['aircraft_id'].tolist()
    previous_solution_table = nearest_flights_selection(previous_solution_table, current_time)

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
            previous_solution_id = previous_solution_table[previous_solution_table['flight_id'] == flight_id]['previous_solution_id'].iloc[0]
            if new_equipment_dict[aircraft_id] not in literal_eval(flight_equipments_subtable[flight_equipments_subtable['flight_id'] == flight_id]['equipment_ids'].iloc[0]):
                disrupted_flights.append({'flight_id': flight_id, 'aircraft_id': aircraft_id, 'previous_solution_id': int(previous_solution_id)})

    return disrupted_flights


def equipment_disrupted_flights_checker(flight_equipments_table: pd.DataFrame,
                                        previous_solution_table: pd.DataFrame,
                                        disruptions_table: pd.DataFrame,
                                        current_time: datetime,
                                        *disruption_ids: int) -> bool:
    """Возвращает True если расписание остается верным(не ломается от смены оснащения ВС, иначе False)"""
    if len(equipment_disrupted_flights(flight_equipments_table,
                                       previous_solution_table,
                                       disruptions_table,
                                       current_time,
                                       *disruption_ids)) > 0:
        return False
    return True


def get_time_from_table(table: pd.DataFrame, previous_solution_id: int, column_name: str) -> datetime:
    """Возвращает объект datetime из таблицы для необходимой строки для departure/arrival(передаётся в column_name)"""
    if 'new' not in column_name:
        previous_solution_id += 1
    string_time = table[table['previous_solution_id'] == previous_solution_id][column_name].iloc[0]

    time_object = parser.parse(string_time, dayfirst=True)
    return time_object


def flight_shift_disrupted_flights(previous_solution_table: pd.DataFrame,
                                   disruptions_table: pd.DataFrame,
                                   *disruption_ids: int) -> list:
    """Возвращает список полётов, которые становятся невыполнимыми из-за переноса рейсов
    (в т.ч. проверка на +50 минут после рейса"""
    problematic_aircrafts = disruptions_table['aircraft_id'].tolist()
    disrupted_flights = []

    for disruption_id in disruption_ids:
        previous_solution_id = disruptions_table[disruptions_table['problematic_flight_shift_id'] == disruption_id]['previous_solution_id'].iloc[0]
        if previous_solution_id not in previous_solution_table['previous_solution_id'].values:
            continue

        new_arrival_time = get_time_from_table(disruptions_table, previous_solution_id, 'new_arrival_time')
        next_departure_time = get_time_from_table(previous_solution_table, previous_solution_id, 'departure_time')

        flight_id = previous_solution_table[previous_solution_table['previous_solution_id'] == previous_solution_id]['flight_id'].iloc[0]
        aircraft_id = problematic_aircrafts[disruption_id]

        time_delta = next_departure_time - new_arrival_time
        # 50 минут = 3000 сек - минимальное окно между рейсами
        if time_delta.total_seconds() < 3000:
            disrupted_flights.append({'flight_id': flight_id, 'aircraft_id': aircraft_id, 'previous_solution_id': int(previous_solution_id)})

    return disrupted_flights


def flight_shift_disrupted_flights_checker(previous_solution_table: pd.DataFrame,
                                           disruptions_table: pd.DataFrame,
                                           *disruption_ids: int) -> bool:
    """Возвращает True если расписание остается верным(не ломается от переноса рейсов, иначе False)"""
    if len(flight_shift_disrupted_flights(previous_solution_table, disruptions_table, *disruption_ids)) > 0:
        return False
    return True


def generate_airport_pairs(previous_solution_table: pd.DataFrame) -> dict:
    """Возвращает словарь, который для каждого ВС хранит данные о каждом рейсе(во вложенных словарях)"""
    departure_time_row = pd.to_datetime(previous_solution_table['departure_time'])
    arrival_time_row = pd.to_datetime(previous_solution_table['arrival_time'])
    departure_airport_row = previous_solution_table['departure_airport_code']
    arrival_airport_row = previous_solution_table['arrival_airport_code']
    flight_id_row = previous_solution_table['flight_id']
    previous_solution_id_row = previous_solution_table['previous_solution_id']

    aircraft_id_row = previous_solution_table['aircraft_id'].astype(str)
    airport_pairs_list = {aircraft_id: [] for aircraft_id in aircraft_id_row.unique()}

    for i in range(len(previous_solution_table)):
        flight_dict = {'departure_time': departure_time_row.iloc[i],
                       'arrival_time': arrival_time_row.iloc[i],
                       'departure_airport': departure_airport_row.iloc[i],
                       'arrival_airport': arrival_airport_row.iloc[i],
                       'flight_id': str(flight_id_row.iloc[i]),
                       'aircraft_id': str(aircraft_id_row.iloc[i]),
                       'previous_solution_id': int(previous_solution_id_row.iloc[i])}
        airport_pairs_list[str(aircraft_id_row.iloc[i])].append(flight_dict)

    return airport_pairs_list


def base_airports_partition(previous_solution_table: pd.DataFrame, *base_airports: str) -> list:
    """Возвращает список связок полётов, разделенных по базовым аэропортам(LED, KJA)"""
    airport_pairs_list = generate_airport_pairs(previous_solution_table)
    partition_list = []
    for aircraft_id in airport_pairs_list.keys():
        current_part = []
        for flight in airport_pairs_list[aircraft_id]:
            current_part.append(flight)
            if flight['arrival_airport'] in base_airports:
                partition_list.append(current_part)
                current_part = []
        if current_part:
            partition_list.append(current_part)
    return partition_list


def extract_trigger_aircraft_ids(disruption_table: pd.DataFrame) -> list:
    """Возвращает список ВС с триггером"""
    aircraft_ids = disruption_table['aircraft_id'].unique()
    return sorted(aircraft_ids.tolist())


def aircraft_flight_line(aircraft_id: int, nearest_schedule: pd.DataFrame) -> pd.DataFrame:
    """Возвращает DataFrame полетов для конкретного ВС"""
    aircraft_flights = nearest_schedule[nearest_schedule['aircraft_id'] == aircraft_id]
    return aircraft_flights


def extract_part_using_flight_id(aircraft_id: int, flight_id: str, airports_partition: list) -> list:
    """Возвращает связку с конкретным рейсом flight_id"""
    trigger_part = []
    for part in airports_partition:
        for flight in part:
            if flight_id == flight['flight_id'] and str(aircraft_id) == flight['aircraft_id']:
                trigger_part = part
    return trigger_part


def extract_part_from_timerange(aircraft_id: int, nearest_schedule: pd.DataFrame, timerange: list, airport_partition: list) -> list:
    """Возвращает связку, один из полётов который попадает в указанный временной отрезок"""
    aircraft_flights = aircraft_flight_line(aircraft_id, nearest_schedule)
    for index, flight in aircraft_flights.iterrows():
        if ((pd.to_datetime(flight['departure_time']) >= timerange[0] and pd.to_datetime(flight['arrival_time']) <= timerange[1])
                or (pd.to_datetime(flight['departure_time']) <= timerange[1] <= pd.to_datetime(flight['arrival_time']))
                or (pd.to_datetime(flight['departure_time']) <= timerange[0] <= pd.to_datetime(flight['arrival_time']))):
            return extract_part_using_flight_id(aircraft_id, flight['flight_id'], airport_partition)


def extract_trigger_time_range(aircraft_id: int, trigger_flight_id: str, airports_partition: list) -> list:
    """Возвращает временной отрезок, в котором находится связка с триггерным рейсом"""
    trigger_part = extract_part_using_flight_id(aircraft_id, trigger_flight_id, airports_partition)
    trigger_departure_time = trigger_part[0]['departure_time']
    trigger_arrival_time = trigger_part[-1]['arrival_time']
    return [trigger_departure_time, trigger_arrival_time]


def from_partition_to_dataframe(partition_list: list) -> pd.DataFrame:
    temp_list = list(chain(*partition_list))
    new_schedule = pd.DataFrame(temp_list)
    return new_schedule


def swap(trigger_aircraft: int,
         health_aircraft: int,
         partition_list: list,
         equipment_disrupted_flights: list,
         flight_shift_disrupted_list: list,
         nearest_flights: pd.DataFrame,
         trigger_aircraft_ids: list,
         trigger_flight_id: str) -> pd.DataFrame:
    """Делаем swap между ВС по связкам"""
    if trigger_aircraft not in trigger_aircraft_ids and health_aircraft in trigger_aircraft_ids:
        trigger_aircraft, health_aircraft, = health_aircraft, trigger_aircraft
    elif ((health_aircraft not in trigger_aircraft_ids and trigger_aircraft not in trigger_aircraft_ids)
          or (health_aircraft in trigger_aircraft_ids and trigger_aircraft in trigger_aircraft_ids)):
        return nearest_flights

    trigger_timerange = extract_trigger_time_range(trigger_aircraft, trigger_flight_id, partition_list)
    trigger_part = extract_part_using_flight_id(trigger_aircraft, trigger_flight_id, partition_list)
    health_part = extract_part_from_timerange(health_aircraft, nearest_flights, trigger_timerange, partition_list)

    for part in partition_list:
        if part == trigger_part:
            for flight in part:
                flight['aircraft_id'] = health_aircraft
        if part == health_part:
            for flight in part:
                flight['aircraft_id'] = trigger_aircraft

    new_schedule = from_partition_to_dataframe(partition_list)
    return new_schedule


def disrupted_flights_for_aircraft_id(aircraft_id: int,
                                      equipment_disrupted_list: list,
                                      flight_shift_disrupted_list: list) -> list:
    """Из двух списков проблемных рейсов получаем список рейсов для конкретного ВС"""
    all_disrupted_flights = equipment_disrupted_list + flight_shift_disrupted_list
    disrupted_flights_for_aircraft_id_list = []
    for flight in all_disrupted_flights:
        if flight['aircraft_id'] == aircraft_id:
            disrupted_flights_for_aircraft_id_list.append(flight)
    return disrupted_flights_for_aircraft_id_list


def flights_objective_function(flights_parts: list, nearest_schedule: pd.DataFrame):
    """Кол-во переставленных связок(рейсов) и суммарное время сдвинутых ТО(не дольше суток) -> min"""


curr_time = datetime(2025, 1, 22, 0, 0)
nearest_sched = nearest_flights_selection(previous_solution, curr_time, 'KJA', 'LED')
# nearest_sched.to_csv('csv_files/nearest_schedule.csv', index=False, sep=';')
parts = base_airports_partition(nearest_sched, 'KJA', 'LED')
b = generate_airport_pairs(nearest_sched)
equipment_disrupted_list = equipment_disrupted_flights(flight_equipments, nearest_sched, problematic_aircraft_equipment, curr_time, 0, 1, 2)
flight_shift_disrupted_list = flight_shift_disrupted_flights(previous_solution, problematic_flight_shift, 0, 1, 2)
disrupted_flights = disrupted_flights_for_aircraft_id(3, equipment_disrupted_list, flight_shift_disrupted_list)
trigger_aircraft_list = (extract_trigger_aircraft_ids(problematic_aircraft_equipment) +
                         extract_trigger_aircraft_ids(problematic_flight_shift))

new_partition = swap(3, 6, parts, equipment_disrupted_list, flight_shift_disrupted_list, nearest_sched, trigger_aircraft_list, 'FV6516')
new_schedule_dataframe = from_partition_to_dataframe(new_partition)
new_schedule_dataframe.to_csv('csv_files/new_schedule.csv', index=False, sep=';')
