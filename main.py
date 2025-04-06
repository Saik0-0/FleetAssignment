import ast
import copy
import itertools
import math
from itertools import chain
import random
import time
import pandas as pd
from ast import literal_eval
from datetime import datetime, timedelta
from dateutil import parser
from matplotlib import pyplot as plt
from gantt_charts import gantt_chart


def change_aircraft_equipment(problematic_aircraft_equipment_table: pd.DataFrame,
                              aircraft_table: pd.DataFrame,
                              *disruption_ids: int) -> pd.DataFrame:
    """Меняет данные по оснащению самолетов в соответствии с таблицей сбоев"""
    problematic_aircraft_equipment_table = problematic_aircraft_equipment_table.loc[[*disruption_ids]]
    problematic_aircrafts = problematic_aircraft_equipment_table['aircraft_id'].tolist()
    new_equipment = {}
    for aircraft in problematic_aircrafts:
        new_equipment[aircraft] = \
            problematic_aircraft_equipment_table[problematic_aircraft_equipment_table['aircraft_id'] == aircraft][
                'equipment_id'].iloc[0]
    for ind in aircraft_table.index:
        if aircraft_table['aircraft_id'].iloc[ind] in problematic_aircrafts:
            aircraft_table.loc[ind, 'equipment_id'] = new_equipment[aircraft_table['aircraft_id'].iloc[ind]]
    return aircraft_table


def change_disrupted_flights_time(problematic_flight_shift_table: pd.DataFrame,
                                  previous_schedule: pd.DataFrame,
                                  base_airports: list,
                                  *disruption_ids: int) -> pd.DataFrame:
    """Меняет данные о времени вылета и прилета рейсов в соответствии с таблицей сбоев"""
    problematic_flight_shift_table = problematic_flight_shift_table.loc[[*disruption_ids]]
    problematic_ids = problematic_flight_shift_table['previous_solution_id'].tolist()
    previous_schedule_parts = base_airports_partition(previous_schedule, *base_airports)
    time_shifts = {}
    for problematic_id in problematic_ids:
        time_shifts[problematic_id] = \
            problematic_flight_shift_table[problematic_flight_shift_table['previous_solution_id'] == problematic_id][
                'shift'].iloc[0]
        time_shifts[problematic_id] = pd.to_timedelta(time_shifts[problematic_id])
    # for ind in previous_schedule.index:
    #     if previous_schedule['previous_solution_id'].iloc[ind] in problematic_ids:
    #         previous_schedule.loc[ind, 'departure_time'] = pd.to_datetime(previous_schedule.loc[ind, 'departure_time'],
    #                                                                       dayfirst=True) + time_shifts[
    #                                                            previous_schedule['previous_solution_id'].iloc[ind]]
    #         previous_schedule.loc[ind, 'arrival_time'] = pd.to_datetime(previous_schedule.loc[ind, 'arrival_time'],
    #                                                                     dayfirst=True) + time_shifts[
    #                                                          previous_schedule['previous_solution_id'].iloc[ind]]
    for part in previous_schedule_parts:
        founded_id = set([flight['previous_solution_id'] for flight in part]) & set(problematic_ids)
        if founded_id:
            time_shift = time_shifts[founded_id.pop()]
            for flight in part:
                ind = previous_schedule[previous_schedule['previous_solution_id'] == flight['previous_solution_id']].index[0]
                previous_schedule.loc[ind, 'departure_time'] = (pd.to_datetime(previous_schedule.loc[ind, 'departure_time'], dayfirst=True)
                                                                + time_shift)
                previous_schedule.loc[ind, 'arrival_time'] = (pd.to_datetime(previous_schedule.loc[ind, 'arrival_time'],
                                                                            dayfirst=True) + time_shift)

    return previous_schedule


pd.set_option("display.max_columns", None)
# парк доступных вс
aircraft = pd.read_csv('csv_files/df_aircraft_with_spare — копия.csv', sep=';')
# авиарейсы
flights = pd.read_csv('csv_files/df_flights.csv', sep=';')
# требования к вс на рейсе
flight_equipments = pd.read_csv('csv_files/df_flight_equipments.csv', sep=';')
# текущее распределение вс на рейсы
previous_solution = pd.read_csv('csv_files/df_previous_solution.csv', sep=';')
previous_solution['arrival_time'] = pd.to_datetime(previous_solution['arrival_time'], format="%d.%m.%Y %H:%M")
previous_solution['departure_time'] = pd.to_datetime(previous_solution['departure_time'], format="%d.%m.%Y %H:%M")
# тех обслуживание вс
technical_service = pd.read_csv('csv_files/df_technical_service.csv', sep=';')

# disruption: изменение оснащения вс
problematic_aircraft_equipment = pd.read_csv('csv_files/df_problematic_aircraft_equipment.csv', sep=';')
# disruption: перенос рейсов
problematic_flight_shift = pd.read_csv('csv_files/df_problematic_flight_shift.csv', sep=';')

# парк доступных вс c измененным оборудованием
updated_aircraft = change_aircraft_equipment(problematic_aircraft_equipment, aircraft, 0, 1, 2, 3, 4)

# updated_aircraft.to_csv('csv_files/new_aircraft.csv', index=False, sep=';')

# updated_previous_solution.to_csv('csv_files/new_previous_solution.csv', index=False, sep=';')


def nearest_flights_selection(previous_solution_table: pd.DataFrame,
                              current_time: datetime,
                              *base_airports: str) -> pd.DataFrame:
    """Возвращает часть расписания (указанная дата + 3 суток)"""
    start = time.time()
    previous_solution_table_result = pd.DataFrame(columns=previous_solution_table.columns)
    for index, flight_row in previous_solution_table.iterrows():
        if current_time <= pd.to_datetime(flight_row['departure_time'], dayfirst=True) <= current_time + timedelta(
                days=2, hours=23, minutes=59):
            if (index + 1 < len(previous_solution_table.index) and pd.to_datetime(
                    previous_solution_table.iloc[index + 1]['departure_time'],
                    dayfirst=True) > current_time + timedelta(days=2, hours=23, minutes=59)
                    and flight_row['arrival_airport_code'] not in base_airports):
                previous_solution_table_result.loc[len(previous_solution_table_result.index)] = flight_row
                sub_index = index + 1
                base_airport_flag = 1
                if sub_index < len(previous_solution_table):
                    while base_airport_flag:
                        if previous_solution_table.iloc[sub_index]['departure_airport_code'] in base_airports:
                            base_airport_flag = 0
                        else:
                            previous_solution_table_result.loc[len(previous_solution_table_result.index)] = \
                                previous_solution_table.iloc[sub_index]
                            if sub_index < len(previous_solution_table) - 1:
                                sub_index += 1
                            else:
                                base_airport_flag = 0
                continue
            if index == 0:
                if flight_row['departure_airport_code'] not in base_airports:
                    continue
            elif (not any((previous_solution_table_result == previous_solution_table.iloc[index - 1]).all(axis=1))
                  and flight_row['departure_airport_code'] not in base_airports):
                continue
            previous_solution_table_result.loc[len(previous_solution_table_result.index)] = flight_row

    return previous_solution_table_result


def equipment_disrupted_flights(flight_equipments_table: pd.DataFrame,
                                nearest_schedule: pd.DataFrame,
                                disruptions_table: pd.DataFrame,
                                current_time: datetime,
                                *disruption_ids: int) -> list:
    """Возвращает список полётов, которые становятся невыполнимыми из-за смены оснащения ВС"""
    disruptions_table = disruptions_table.loc[[*disruption_ids]]
    problematic_aircrafts = disruptions_table['aircraft_id'].tolist()

    # Структура словаря: ключи - id самолетов с изменением оборудования,
    # значения - индексы нового оборудования
    new_equipment_dict = {}
    for aircraft_id in problematic_aircrafts:
        new_equipment_dict[aircraft_id] = \
            disruptions_table[disruptions_table['aircraft_id'] == aircraft_id]['equipment_id'].tolist()[0]

    problematic_flights = nearest_schedule[nearest_schedule['aircraft_id'].isin(problematic_aircrafts)]

    # Структура словаря: ключи - id самолетов с изменением оборудования,
    # значения - set из id рейсов куда распределены самолеты
    problematic_flights_id = {}
    for aircraft_id in problematic_aircrafts:
        problematic_flights_id[aircraft_id] = problematic_flights[problematic_flights['aircraft_id'] == aircraft_id][
            ['flight_id', 'previous_solution_id']]

    disrupted_flights = []
    for aircraft_id in problematic_aircrafts:
        flight_equipments_subtable = flight_equipments_table[
            flight_equipments_table['flight_id'].isin(problematic_flights_id[aircraft_id]['flight_id'].tolist())]
        for _, flight_id in problematic_flights_id[aircraft_id].iterrows():
            if new_equipment_dict[aircraft_id] not in literal_eval(
                    flight_equipments_subtable[flight_equipments_subtable['flight_id'] == flight_id['flight_id']][
                        'equipment_ids'].iloc[0]):
                disrupted_flights.append({'flight_id': flight_id['flight_id'], 'aircraft_id': aircraft_id,
                                          'previous_solution_id': int(flight_id['previous_solution_id'])})

    return disrupted_flights


def equipment_disrupted_flights_checker(flight_equipments_table: pd.DataFrame,
                                        nearest_schedule: pd.DataFrame,
                                        disruptions_table: pd.DataFrame,
                                        current_time: datetime,
                                        *disruption_ids: int) -> bool:
    """Возвращает True если расписание остается верным(не ломается от смены оснащения ВС, иначе False)"""
    if len(equipment_disrupted_flights(flight_equipments_table, nearest_schedule, disruptions_table,
                                       current_time, *disruption_ids)) > 0:
        return False
    return True


def get_time_from_table(table: pd.DataFrame, previous_solution_id: int, column_name: str) -> datetime:
    """Возвращает объект datetime из таблицы для необходимой строки для departure/arrival(передаётся в column_name)"""
    if 'new' not in column_name:
        previous_solution_id += 1
    string_time = table[table['previous_solution_id'] == previous_solution_id][column_name].iloc[0]
    # print(string_time, type(string_time))
    if type(string_time) == str:
        time_object = parser.parse(string_time, dayfirst=True)
    else:
        return string_time
    return time_object


def flight_shift_disrupted_flights(nearest_schedule: pd.DataFrame,
                                   disruptions_table: pd.DataFrame,
                                   *disruption_ids: int) -> list:
    """Возвращает список полётов, которые становятся невыполнимыми из-за переноса рейсов
    (в т.ч. проверка на +50 минут после рейса"""
    problematic_aircrafts = disruptions_table['aircraft_id'].tolist()
    disrupted_flights = []

    for disruption_id in disruption_ids:
        previous_solution_id = disruptions_table[disruptions_table['problematic_flight_shift_id'] == disruption_id][
            'previous_solution_id'].iloc[0]
        if previous_solution_id not in nearest_schedule['previous_solution_id'].values:
            continue

        new_arrival_time = get_time_from_table(disruptions_table, previous_solution_id, 'new_arrival_time')
        next_departure_time = get_time_from_table(nearest_schedule, previous_solution_id, 'departure_time')

        flight_id = nearest_schedule[nearest_schedule['previous_solution_id'] == previous_solution_id][
            'flight_id'].iloc[0]
        aircraft_id = problematic_aircrafts[disruption_id]

        time_delta = next_departure_time - new_arrival_time
        # 30 минут = минимальное окно между рейсами
        if time_delta.total_seconds() < 30 * 60:
            disrupted_flights.append(
                {'flight_id': flight_id, 'aircraft_id': aircraft_id, 'previous_solution_id': int(previous_solution_id)})

    return disrupted_flights


def flight_shift_disrupted_flights_checker(nearest_schedule: pd.DataFrame,
                                           disruptions_table: pd.DataFrame,
                                           *disruption_ids: int) -> bool:
    """Возвращает True если расписание остается верным(не ломается от переноса рейсов, иначе False)"""
    if len(flight_shift_disrupted_flights(nearest_schedule, disruptions_table, *disruption_ids)) > 0:
        return False
    return True


def generate_airport_pairs(previous_solution_table: pd.DataFrame) -> dict:
    """Возвращает словарь, который для каждого ВС хранит данные о каждом рейсе(во вложенных словарях)"""
    departure_time_row = pd.to_datetime(previous_solution_table['departure_time'])
    arrival_time_row = pd.to_datetime(previous_solution_table['arrival_time'])
    departure_airport_row = previous_solution_table['departure_airport_code']
    arrival_airport_row = previous_solution_table['arrival_airport_code']
    aircraft_type_row = previous_solution_table['aircraft_type']
    flight_id_row = previous_solution_table['flight_id']
    previous_solution_id_row = previous_solution_table['previous_solution_id']

    aircraft_id_row = previous_solution_table['aircraft_id'].astype(str)
    airport_pairs_list = {aircraft_id: [] for aircraft_id in aircraft_id_row.unique()}

    for i in range(len(previous_solution_table)):
        flight_dict = {'departure_time': departure_time_row.iloc[i],
                       'arrival_time': arrival_time_row.iloc[i],
                       'flight_id': str(flight_id_row.iloc[i]),
                       'departure_airport_code': departure_airport_row.iloc[i],
                       'arrival_airport_code': arrival_airport_row.iloc[i],
                       'aircraft_type': aircraft_type_row.iloc[i],
                       'aircraft_id': int(aircraft_id_row.iloc[i]),
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
            if flight['arrival_airport_code'] in base_airports:
                partition_list.append(current_part)
                current_part = []
        if current_part:
            partition_list.append(current_part)
    return partition_list


def extract_trigger_aircraft_ids(disruption_table: pd.DataFrame) -> list:
    """Возвращает список ВС с триггером"""
    aircraft_ids = disruption_table['aircraft_id'].unique()
    return sorted(aircraft_ids.tolist())


def disrupted_flights_for_aircraft_id(aircraft_id: int,
                                      equipment_disrupted_list: list,
                                      flight_shift_disrupted_list: list) -> list:
    """Из двух списков проблемных рейсов получаем список рейсов для конкретного ВС"""
    all_disrupted_flights = equipment_disrupted_list + flight_shift_disrupted_list
    disrupted_flights_for_aircraft_id_list = []
    for flight in all_disrupted_flights:
        if flight['aircraft_id'] == aircraft_id:
            disrupted_flights_for_aircraft_id_list.append(flight)
    # print(f'Disrupted flights for aircraft {aircraft_id} is {disrupted_flights_for_aircraft_id_list}')
    return disrupted_flights_for_aircraft_id_list


def all_trigger_aircraft_and_flight(equipment_disruption_table: pd.DataFrame,
                                    time_shifts_disruption_table: pd.DataFrame,
                                    old_flight_equipment_table: pd.DataFrame,
                                    old_nearest_previous_schedule: pd.DataFrame,
                                    current_time: datetime) -> dict:
    """Возвращает словарь с ключами id ВС с триггером и в значениях рейсы которые не могут быть выполнены"""
    equipment_trigger_aircrafts = extract_trigger_aircraft_ids(equipment_disruption_table)
    time_shifts_trigger_aircrafts = extract_trigger_aircraft_ids(time_shifts_disruption_table)

    trigger_aircrafts = list(set(equipment_trigger_aircrafts + time_shifts_trigger_aircrafts))
    equipment_trigger_flights = equipment_disrupted_flights(old_flight_equipment_table, old_nearest_previous_schedule,
                                                            equipment_disruption_table, current_time,
                                                            *range(len(equipment_disruption_table)))

    time_shifts_trigger_flights = flight_shift_disrupted_flights(old_nearest_previous_schedule,
                                                                 time_shifts_disruption_table,
                                                                 *range(len(time_shifts_disruption_table)))

    trigger_flights_dict = {}
    for aircraft in trigger_aircrafts:
        trigger_flights_dict[aircraft] = disrupted_flights_for_aircraft_id(aircraft,
                                                                           equipment_trigger_flights,
                                                                           time_shifts_trigger_flights)

    return trigger_flights_dict


def aircraft_and_flight_random_choice(trigger_flights_dict: dict) -> tuple:
    """Случайным образом выбирает триггерный ВС и рейс для него"""
    trigger_aircraft = random.choice(list(trigger_flights_dict.keys()))
    while trigger_flights_dict[trigger_aircraft] == []:
        trigger_aircraft = random.choice(list(trigger_flights_dict.keys()))
    trigger_flight = random.choice(trigger_flights_dict[trigger_aircraft])
    return trigger_aircraft, trigger_flight


def aircraft_for_swap_random_choice(aircraft_table: pd.DataFrame, trigger_aircraft: int) -> int:
    """Случайно выбирает второе ВС с которым будем делать swap"""
    table_for_second_aircraft = aircraft_table[(aircraft_table['aircraft_id'] != trigger_aircraft)
                                               & (aircraft_table['equipment_id'] != 0)]['aircraft_id']
    return random.choice(table_for_second_aircraft.unique())


def aircraft_flight_line(aircraft_id: int, nearest_schedule: pd.DataFrame) -> pd.DataFrame:
    """Возвращает DataFrame полетов для конкретного ВС"""
    aircraft_flights = nearest_schedule[nearest_schedule['aircraft_id'] == aircraft_id].reset_index(drop=True)
    return aircraft_flights


def extract_part_using_flight_id(aircraft_id: int, flight_id: dict, airports_partition: list) -> list:
    """Возвращает связку с конкретным рейсом flight_id"""
    trigger_part = []
    for part in airports_partition:
        for flight in part:
            if (flight_id['flight_id'] == flight['flight_id']
                    and flight_id['previous_solution_id'] == flight['previous_solution_id']
                    and aircraft_id == flight['aircraft_id']):
                trigger_part = part
    return trigger_part


def extract_next_or_previous_part(aircraft_id: int, part: list, airports_partition: list, param: str) -> list:
    """Возвращает предыдущую или следующую связку ВС от переданной связки.
    Параметр flag принимает строки 'prev' или 'next'"""
    for index in range(len(airports_partition)):
        if airports_partition[index] == part:
            if param == 'prev' and index > 0:
                if airports_partition[index - 1][0]['aircraft_id'] == aircraft_id:
                    return airports_partition[index - 1]
            if param == 'next' and index < len(airports_partition) - 1:
                if airports_partition[index + 1][0]['aircraft_id'] == aircraft_id:
                    return airports_partition[index + 1]
    return []


def extract_part_from_timerange(aircraft_id: int, nearest_schedule: pd.DataFrame, timerange: list,
                                airport_partition: list) -> list:
    """Возвращает связку, один из полётов который попадает в указанный временной отрезок"""
    aircraft_flights = aircraft_flight_line(aircraft_id, nearest_schedule)
    for index, flight in aircraft_flights.iterrows():
        if ((pd.to_datetime(flight['departure_time'], dayfirst=True) >= timerange[0] and pd.to_datetime(
                flight['arrival_time'], dayfirst=True) <= timerange[1])
                or (pd.to_datetime(flight['departure_time'], dayfirst=True) <= timerange[1] <= pd.to_datetime(
                    flight['arrival_time'], dayfirst=True))
                or (pd.to_datetime(flight['departure_time'], dayfirst=True) <= timerange[0] <= pd.to_datetime(
                    flight['arrival_time'], dayfirst=True))):
            return extract_part_using_flight_id(aircraft_id, flight.to_dict(), airport_partition)
    return []


def extract_trigger_time_range(aircraft_id: int, trigger_flight_id: dict, airports_partition: list) -> list:
    """Возвращает временной отрезок, в котором находится связка с триггерным рейсом"""
    trigger_part = extract_part_using_flight_id(aircraft_id, trigger_flight_id, airports_partition)
    trigger_departure_time = trigger_part[0]['departure_time']
    trigger_arrival_time = trigger_part[-1]['arrival_time']
    return [trigger_departure_time, trigger_arrival_time]


def schedule_sort(partition_list: list) -> list:
    """
        Сортирует список связок:
        1. Внутри каждой связки сортирует рейсы по 'departure_time'.
        2. Сортирует сами связки: сначала по 'aircraft_id' (первого рейса в цепочке),
           а затем по самому раннему 'departure_time' в цепочке.
        """
    partition_list_copy = copy.deepcopy(partition_list)
    # for chain_flights in partition_list_copy:
    #     chain_flights.sort(key=lambda x: x['departure_time'])

    partition_list_copy.sort(key=lambda chain_flights: (chain_flights[0]['aircraft_id'],
                                                        chain_flights[0]['departure_time']))
    return partition_list_copy


def from_partition_to_dataframe(partition_list: list) -> pd.DataFrame:
    """Из списка связок получаем расписание в DataFrame"""
    sorted_partition = schedule_sort(partition_list)
    temp_list = list(chain(*sorted_partition))
    new_schedule = pd.DataFrame(temp_list)
    new_schedule['departure_time'] = new_schedule['departure_time'].astype(str)
    new_schedule['arrival_time'] = new_schedule['arrival_time'].astype(str)

    return new_schedule


def get_time_diapason_for_aircraft(aircraft_id: int, schedule: pd.DataFrame) -> dict:
    """Извлекает начальный момент времени и конечный момент времени задействования самолёта в расписании"""
    aircraft_flights = aircraft_flight_line(aircraft_id, schedule)
    min_time = min(pd.to_datetime(aircraft_flights['departure_time']))
    max_time = max(pd.to_datetime(aircraft_flights['arrival_time']))
    return {'min_time': min_time, 'max_time': max_time}


def remake_technical_service_table(technical_service_table: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Извлекает из таблицы тех обслуживания строки, попадающие в нужный диапазон времени"""
    technical_service_table_copy = pd.DataFrame(columns=technical_service_table.columns)
    prev_aircraft_id = technical_service_table['aircraft_id'].iloc[0]
    time_diapason = get_time_diapason_for_aircraft(prev_aircraft_id, schedule)
    for index, row in technical_service_table.iterrows():
        aircraft_id = row['aircraft_id']
        if aircraft_id != prev_aircraft_id:
            time_diapason = get_time_diapason_for_aircraft(aircraft_id, schedule)
        if (pd.to_datetime(row['time_finish'], dayfirst=True) <= time_diapason['max_time']
                and pd.to_datetime(row['time_start'], dayfirst=True) >= time_diapason['min_time']):
            technical_service_table_copy.loc[len(technical_service_table_copy.index)] = row
    return technical_service_table_copy


def allowed_time_differences(aircraft_flights: pd.DataFrame, time_size: timedelta, allowed_times: list, current_time: datetime):
    """Проверяет, есть ли необходимый диапазон времени в перерывах между переданными рейсами конкретного ВС,
    в параметр allowed_times добавляются временные промежутки когда может быть проведено ТО"""
    flag = False
    if len(aircraft_flights) > 0:
        if pd.to_datetime(aircraft_flights['departure_time']).iloc[0] - current_time >= time_size:
            flag = True
            allowed_times.append([current_time, pd.to_datetime(aircraft_flights['departure_time']).iloc[0]])
    for index in range(len(aircraft_flights.index) - 1):
        arrival_time = pd.to_datetime(aircraft_flights['arrival_time']).iloc[index]
        next_departure_time = pd.to_datetime(aircraft_flights['departure_time']).iloc[index + 1]
        time_delta = next_departure_time - arrival_time
        if time_delta >= time_size:
            allowed_times.append([arrival_time, next_departure_time])
            flag = True
    if len(aircraft_flights) > 0:
        last_arrival_time = pd.to_datetime(aircraft_flights['arrival_time']).iloc[-1]
        if current_time + timedelta(days=2, hours=23, minutes=59) - last_arrival_time >= time_size:
            flag = True
            allowed_times.append([last_arrival_time, current_time + timedelta(days=2, hours=23, minutes=59)])
    if flag is False:
        print(aircraft_flights)
        print(time_size)
        print(allowed_times)
    return flag


def checking_allowed_ts_time(technical_service_table: pd.DataFrame, ts_allowed_time: list, ts_used_time: list) -> int:
    ts_times = list(zip(pd.to_datetime(technical_service_table['time_start'], dayfirst=True),
                        pd.to_datetime(technical_service_table['time_finish'], dayfirst=True)))
    flag = False
    count = 0
    for allowed_time in ts_allowed_time:
        for ts_time in ts_times:
            if ((allowed_time[0] >= ts_time[0] and allowed_time[1] <= ts_time[1]) or
                    (ts_time[0] >= allowed_time[0] and ts_time[1] <= allowed_time[1])):
                count += 1
                if allowed_time not in ts_used_time:
                    flag = True
                    ts_allowed_time.remove(allowed_time)
                    ts_used_time.append(allowed_time)
                    break
    # Штраф если flag = False
    if not flag:
        count = 1000
    if count >= 1000:
        print(ts_allowed_time)
        print(ts_times)
    return count


def move_parts(trigger_aircraft: int, health_aircraft: int, trigger_part: list, health_part: list,
               partition_list: list) -> list:
    """Меняет местами назначения для переданных ВС в переданных связках, возвращает новое разбиение на связки"""
    partition_list_copy = copy.deepcopy(partition_list)
    for part in partition_list_copy:
        if part == trigger_part:
            for flight in part:
                flight['aircraft_id'] = health_aircraft
        if part == health_part:
            for flight in part:
                flight['aircraft_id'] = trigger_aircraft
    return partition_list_copy


def smart_swap(trigger_aircraft: int,
               health_aircraft: int,
               partition_list: list,
               nearest_flights: pd.DataFrame,
               trigger_flight_id: dict,
               flight_equipment_table: pd.DataFrame,
               aircraft_equipment_table: pd.DataFrame,
               swapped_flag: list) -> tuple:
    """Делаем swap между ВС по связкам пока не будут удовлетворены все штрафные функции"""
    dict_of_swapped_flights = {'was_triggered': [], 'was_health': []}
    health_aircraft_info = aircraft_equipment_table[aircraft_equipment_table['aircraft_id'] == health_aircraft]
    trigger_aircraft_eq_info = \
        aircraft_equipment_table[aircraft_equipment_table['aircraft_id'] == trigger_aircraft]['equipment_id'].iloc[0]
    trigger_part = extract_part_using_flight_id(trigger_aircraft, trigger_flight_id, partition_list)
    if trigger_part == []:
        return from_partition_to_dataframe(partition_list), dict_of_swapped_flights
    trigger_timerange = extract_trigger_time_range(trigger_aircraft, trigger_flight_id, partition_list)
    health_part = extract_part_from_timerange(health_aircraft, nearest_flights, trigger_timerange, partition_list)

    temp_trigger_part_for_find_prev = trigger_part
    temp_health_part_for_find_prev = health_part
    temp_trigger_part_for_find_next = trigger_part
    temp_health_part_for_find_next = health_part

    health_prev_part = extract_next_or_previous_part(health_aircraft, temp_health_part_for_find_prev,
                                                     partition_list, 'prev')
    health_next_part = extract_next_or_previous_part(health_aircraft, temp_health_part_for_find_next,
                                                     partition_list, 'next')

    if (health_part == [] and (health_prev_part == [] or health_next_part == [])
            and health_aircraft_info['reserve_q'].iloc[0] is False):
        return from_partition_to_dataframe(partition_list), dict_of_swapped_flights

    if trigger_aircraft_eq_info == 0 and health_part != []:
        return from_partition_to_dataframe(partition_list), dict_of_swapped_flights

    health_equipment_flag = True
    trigger_equipment_flag = True

    for flight in trigger_part:
        flight_info = flight_equipment_table[flight_equipment_table['flight_id'] == flight['flight_id']]['equipment_ids'].iloc[0]
        if health_aircraft_info['equipment_id'].iloc[0] not in literal_eval(flight_info):
            health_equipment_flag = False

    for flight in health_part:
        flight_info = flight_equipment_table[flight_equipment_table['flight_id'] == flight['flight_id']]['equipment_ids'].iloc[0]
        if trigger_aircraft_eq_info not in literal_eval(flight_info):
            trigger_equipment_flag = False

    if not trigger_equipment_flag or not health_equipment_flag:
        return from_partition_to_dataframe(partition_list), dict_of_swapped_flights

    if (aircraft_equipment_table[aircraft_equipment_table['aircraft_id'] == trigger_aircraft]['equipment_id'].iloc[0] == 0 and health_part != []):
        return from_partition_to_dataframe(partition_list), dict_of_swapped_flights

    temp_partition = move_parts(trigger_aircraft, health_aircraft, trigger_part, health_part, partition_list)
    swapped_flag[0] = True
    dict_of_swapped_flights['was_triggered'].append(trigger_part)
    dict_of_swapped_flights['was_health'].append(health_part)

    pd_for_test = from_partition_to_dataframe(temp_partition)
    pd_trigger_for_test = pd_for_test.loc[(pd_for_test['aircraft_id'] == trigger_aircraft)]
    pd_health_for_test = pd_for_test.loc[(pd_for_test['aircraft_id'] == health_aircraft)]

    # Сюда: добавить функцию проверки insert

    empty_prev_flag = False
    good_prev_partition = False
    empty_next_flag = False
    good_next_partition = False
    equipment_prev_flag = True
    equipment_next_flag = True

    while True:
        trigger_prev_part = extract_next_or_previous_part(trigger_aircraft, temp_trigger_part_for_find_prev,
                                                          partition_list, 'prev')
        health_prev_part = extract_next_or_previous_part(health_aircraft, temp_health_part_for_find_prev,
                                                         partition_list, 'prev')
        if (trigger_prev_part != [] and health_prev_part != []
                and good_prev_partition is False and empty_prev_flag is False and equipment_prev_flag is True):
            temp_partition_without_check = move_parts(trigger_aircraft, health_aircraft, trigger_prev_part,
                                                      health_prev_part, temp_partition)

            pd_for_test = from_partition_to_dataframe(temp_partition_without_check)
            pd_trigger_for_test = pd_for_test.loc[(pd_for_test['aircraft_id'] == trigger_aircraft)]
            pd_health_for_test = pd_for_test.loc[(pd_for_test['aircraft_id'] == health_aircraft)]
            prev_penalty_flag = (penalty_function_for_swap(pd_trigger_for_test, trigger_aircraft, trigger_prev_part,
                                                           health_prev_part, 'prev', flight_equipment_table,
                                                           aircraft_equipment_table) == 0
                                 and penalty_function_for_swap(pd_health_for_test, health_aircraft, health_prev_part,
                                                               trigger_prev_part, 'prev', flight_equipment_table,
                                                               aircraft_equipment_table) == 0)
            equipment_prev_flag = (equipment_checker_in_swap(pd_trigger_for_test,
                                                             trigger_aircraft,
                                                             trigger_prev_part,
                                                             health_prev_part,
                                                             'prev',
                                                             flight_equipment_table,
                                                             aircraft_equipment_table)
                                   and equipment_checker_in_swap(pd_health_for_test,
                                                                 health_aircraft,
                                                                 health_prev_part,
                                                                 trigger_prev_part,
                                                                 'prev',
                                                                 flight_equipment_table,
                                                                 aircraft_equipment_table))
            if equipment_prev_flag:
                temp_partition = temp_partition_without_check
                temp_trigger_part_for_find_prev = trigger_prev_part
                temp_health_part_for_find_prev = health_prev_part
                dict_of_swapped_flights['was_triggered'].append(trigger_prev_part)
                dict_of_swapped_flights['was_health'].append(health_prev_part)
            if prev_penalty_flag:
                good_prev_partition = True
                temp_partition = temp_partition_without_check
                temp_trigger_part_for_find_prev = trigger_prev_part
                temp_health_part_for_find_prev = health_prev_part
                dict_of_swapped_flights['was_triggered'].append(trigger_prev_part)
                dict_of_swapped_flights['was_health'].append(health_prev_part)
        elif trigger_prev_part == [] or health_prev_part == []:
            empty_prev_flag = True

        trigger_next_part = extract_next_or_previous_part(trigger_aircraft, temp_trigger_part_for_find_next,
                                                          partition_list, 'next')
        health_next_part = extract_next_or_previous_part(health_aircraft, temp_health_part_for_find_next,
                                                         partition_list, 'next')
        if (trigger_next_part != [] and health_next_part != []
                and good_next_partition is False and empty_next_flag is False and equipment_next_flag is True):
            temp_partition_without_check = move_parts(trigger_aircraft, health_aircraft, trigger_next_part,
                                                      health_next_part, temp_partition)

            pd_for_test = from_partition_to_dataframe(temp_partition_without_check)
            pd_trigger_for_test = pd_for_test.loc[(pd_for_test['aircraft_id'] == trigger_aircraft)]
            pd_health_for_test = pd_for_test.loc[(pd_for_test['aircraft_id'] == health_aircraft)]
            next_penalty_flag = (penalty_function_for_swap(pd_trigger_for_test, trigger_aircraft, trigger_next_part,
                                                           health_next_part, 'next', flight_equipment_table,
                                                           aircraft_equipment_table) == 0
                                 and penalty_function_for_swap(pd_health_for_test, health_aircraft, health_next_part,
                                                               trigger_next_part, 'next', flight_equipment_table,
                                                               aircraft_equipment_table) == 0)
            equipment_next_flag = (equipment_checker_in_swap(pd_trigger_for_test,
                                                             trigger_aircraft,
                                                             trigger_next_part,
                                                             health_next_part,
                                                             'next',
                                                             flight_equipment_table,
                                                             aircraft_equipment_table)
                                   and equipment_checker_in_swap(pd_health_for_test,
                                                                 health_aircraft,
                                                                 health_next_part,
                                                                 trigger_next_part,
                                                                 'next',
                                                                 flight_equipment_table,
                                                                 aircraft_equipment_table))
            if equipment_next_flag:
                temp_partition = temp_partition_without_check
                temp_trigger_part_for_find_next = trigger_next_part
                temp_health_part_for_find_next = health_next_part
                dict_of_swapped_flights['was_triggered'].append(trigger_prev_part)
                dict_of_swapped_flights['was_health'].append(health_prev_part)
            if next_penalty_flag:
                good_next_partition = True
                temp_partition = temp_partition_without_check
                temp_trigger_part_for_find_next = trigger_next_part
                temp_health_part_for_find_next = health_next_part
                dict_of_swapped_flights['was_triggered'].append(trigger_prev_part)
                dict_of_swapped_flights['was_health'].append(health_prev_part)
        elif trigger_next_part == [] or health_next_part == []:
            empty_next_flag = True

        if not equipment_next_flag or not equipment_prev_flag:
            break

        if good_next_partition and good_prev_partition:
            break

        if empty_next_flag is True and empty_prev_flag is True:
            break

        if (good_prev_partition is True and empty_next_flag is True) or (
                good_next_partition is True and empty_prev_flag is True):
            break
    print(f'swapped')
    return from_partition_to_dataframe(temp_partition), dict_of_swapped_flights


def swap(trigger_aircraft: int,
         health_aircraft: int,
         partition_list: list,
         nearest_flights: pd.DataFrame,
         trigger_aircraft_ids: list,
         trigger_flight_id: dict) -> pd.DataFrame:
    """Делаем swap между ВС по связкам"""
    if trigger_aircraft not in trigger_aircraft_ids and health_aircraft in trigger_aircraft_ids:
        trigger_aircraft, health_aircraft, = health_aircraft, trigger_aircraft
    elif ((health_aircraft not in trigger_aircraft_ids and trigger_aircraft not in trigger_aircraft_ids)
          or (health_aircraft in trigger_aircraft_ids and trigger_aircraft in trigger_aircraft_ids)):
        return nearest_flights

    trigger_timerange = extract_trigger_time_range(trigger_aircraft, trigger_flight_id, partition_list)
    trigger_part = extract_part_using_flight_id(trigger_aircraft, trigger_flight_id, partition_list)
    health_part = extract_part_from_timerange(health_aircraft, nearest_flights, trigger_timerange, partition_list)

    new_partition = move_parts(trigger_aircraft, health_aircraft, trigger_part, health_part, partition_list)

    new_schedule = from_partition_to_dataframe(new_partition)
    return new_schedule


def schedule_time_checker_in_swap(new_schedule: pd.DataFrame,
                                  aircraft_id: int,
                                  previous_part: list,
                                  moved_part: list,
                                  next_or_prev_flag: str) -> bool:
    """Функция для проверки переставленной в swap части на временные накладки"""
    aircraft_flights = aircraft_flight_line(aircraft_id, new_schedule)
    flag = True
    if next_or_prev_flag == 'prev':
        if (previous_part[0]['previous_solution_id'] - 1) in aircraft_flights['previous_solution_id'].tolist():
            index_for_check_first = aircraft_flights[
                aircraft_flights['previous_solution_id'] == previous_part[0]['previous_solution_id'] - 1].index[0]
            # print(moved_part)
            # print(moved_part[-1]['previous_solution_id'])
            # print(aircraft_flights)
            index_for_check_last = \
                aircraft_flights[
                    aircraft_flights['previous_solution_id'] == moved_part[-1]['previous_solution_id']].index[
                    0]
            for index in range(index_for_check_first, index_for_check_last + 1):
                if index + 1 in aircraft_flights.index:
                    arrival_time = aircraft_flights['arrival_time'].iloc[index]
                    next_departure_time = aircraft_flights['departure_time'].iloc[index + 1]
                    if (pd.to_datetime(next_departure_time) - pd.to_datetime(arrival_time)).total_seconds() < 30 * 60:
                        flag = False
    elif next_or_prev_flag == 'next':
        if previous_part[-1]['previous_solution_id'] + 1 in aircraft_flights['previous_solution_id'].tolist():
            index_for_check_last = aircraft_flights[
                aircraft_flights['previous_solution_id'] == previous_part[-1]['previous_solution_id'] + 1].index[0]
            index_for_check_first = \
                aircraft_flights[
                    aircraft_flights['previous_solution_id'] == moved_part[0]['previous_solution_id']].index[0]
            for index in range(index_for_check_first, index_for_check_last + 1):
                if index + 1 in aircraft_flights.index:
                    arrival_time = aircraft_flights['arrival_time'].iloc[index]
                    next_departure_time = aircraft_flights['departure_time'].iloc[index + 1]
                    if (pd.to_datetime(next_departure_time) - pd.to_datetime(arrival_time)).total_seconds() < 30 * 60:
                        flag = False
    return flag


def airport_checker_in_swap(new_schedule: pd.DataFrame,
                            aircraft_id: int,
                            previous_part: list,
                            moved_part: list,
                            next_or_prev_flag: str) -> bool:
    """Функция для проверки переставленной в swap части на совпадение по аэропортам"""
    aircraft_flights = aircraft_flight_line(aircraft_id, new_schedule)
    flag = True
    if next_or_prev_flag == 'prev':
        if previous_part[0]['previous_solution_id'] - 1 in aircraft_flights['previous_solution_id'].tolist():
            index_for_check_first = aircraft_flights[
                aircraft_flights['previous_solution_id'] == previous_part[0]['previous_solution_id'] - 1].index[0]
            index_for_check_last = \
                aircraft_flights[
                    aircraft_flights['previous_solution_id'] == moved_part[-1]['previous_solution_id']].index[
                    0]
            for index in range(index_for_check_first, index_for_check_last + 1):
                if index + 1 in aircraft_flights.index:
                    arrival_airport = aircraft_flights['arrival_airport_code'].iloc[index]
                    next_departure_airport = aircraft_flights['departure_airport_code'].iloc[index + 1]
                    if arrival_airport != next_departure_airport:
                        flag = False
    elif next_or_prev_flag == 'next':
        if previous_part[-1]['previous_solution_id'] + 1 in aircraft_flights['previous_solution_id'].tolist():
            index_for_check_last = aircraft_flights[
                aircraft_flights['previous_solution_id'] == previous_part[-1]['previous_solution_id'] + 1].index[0]
            index_for_check_first = \
                aircraft_flights[
                    aircraft_flights['previous_solution_id'] == moved_part[0]['previous_solution_id']].index[0]
            for index in range(index_for_check_first, index_for_check_last + 1):
                if index + 1 in aircraft_flights.index:
                    arrival_airport = aircraft_flights['arrival_airport_code'].iloc[index]
                    next_departure_airport = aircraft_flights['departure_airport_code'].iloc[index + 1]
                    if arrival_airport != next_departure_airport:
                        flag = False
    return flag


def equipment_checker_in_swap(new_schedule: pd.DataFrame,
                              aircraft_id: int,
                              previous_part: list,
                              moved_part: list,
                              next_or_prev_flag: str,
                              flight_equipment_table: pd.DataFrame,
                              aircraft_equipment_table: pd.DataFrame) -> bool:
    """Функция для проверки переставленной части в swap на совпадение по оснащению самолетов"""
    aircraft_flights = aircraft_flight_line(aircraft_id, new_schedule)
    aircraft_equipment = \
        aircraft_equipment_table[aircraft_equipment_table['aircraft_id'] == aircraft_id]['equipment_id'].iloc[0]
    flag = True
    if next_or_prev_flag == 'prev':
        if previous_part[0]['previous_solution_id'] - 1 in aircraft_flights['previous_solution_id'].tolist():
            index_for_check_first = aircraft_flights[
                aircraft_flights['previous_solution_id'] == previous_part[0]['previous_solution_id'] - 1].index[0]
        else:
            index_for_check_first = aircraft_flights[
                aircraft_flights['previous_solution_id'] == moved_part[0]['previous_solution_id']].index[0]
        index_for_check_last = \
            aircraft_flights[
                aircraft_flights['previous_solution_id'] == moved_part[-1]['previous_solution_id']].index[
                0]
        for index, flight_id in aircraft_flights['flight_id'].items():
            if index_for_check_first <= index <= index_for_check_last:
                flight_equipment = \
                    flight_equipment_table[flight_equipment_table['flight_id'] == flight_id]['equipment_ids'].iloc[
                        0]
                flight_equipment = list(ast.literal_eval(flight_equipment))
                if aircraft_equipment not in flight_equipment:
                    flag = False
    elif next_or_prev_flag == 'next':
        if previous_part[-1]['previous_solution_id'] + 1 in aircraft_flights['previous_solution_id'].tolist():
            index_for_check_last = aircraft_flights[
                aircraft_flights['previous_solution_id'] == previous_part[-1]['previous_solution_id'] + 1].index[0]
        else:
            index_for_check_last = aircraft_flights[
                aircraft_flights['previous_solution_id'] == moved_part[-1]['previous_solution_id']].index[0]
        index_for_check_first = \
            aircraft_flights[
                aircraft_flights['previous_solution_id'] == moved_part[0]['previous_solution_id']].index[0]
        for index, flight_id in aircraft_flights['flight_id'].items():
            if index_for_check_first <= index <= index_for_check_last:
                flight_equipment = \
                    flight_equipment_table[flight_equipment_table['flight_id'] == flight_id]['equipment_ids'].iloc[
                        0]
                flight_equipment = list(ast.literal_eval(flight_equipment))
                if aircraft_equipment not in flight_equipment:
                    flag = False
    return flag


def penalty_function_for_swap(new_schedule: pd.DataFrame,
                              aircraft_id: int,
                              previous_part: list,
                              moved_part: list,
                              next_or_prev_flag: str,
                              flight_equipment_table: pd.DataFrame,
                              aircraft_equipment_table: pd.DataFrame) -> int:
    """Суммарная штрафная функция для проверки swap"""
    penalty = 1000
    if not (schedule_time_checker_in_swap(new_schedule, aircraft_id, previous_part, moved_part, next_or_prev_flag)
            and airport_checker_in_swap(new_schedule, aircraft_id, previous_part, moved_part, next_or_prev_flag)):
        # and equipment_checker_in_swap(new_schedule, aircraft_id, previous_part, moved_part, next_or_prev_flag,
        #                               flight_equipment_table, aircraft_equipment_table)):
        return penalty
    return 0


def schedule_differences(previous_schedule: pd.DataFrame, new_schedule: pd.DataFrame) -> int:
    """Кол-во переставленных рейсов -> min"""
    prev_flights = []
    new_flights = []
    diff = 0
    for index in range(len(previous_schedule)):
        prev_flights.append(
            (previous_schedule['aircraft_id'].iloc[index], previous_schedule['previous_solution_id'].iloc[index]))
        new_flights.append((new_schedule['aircraft_id'].iloc[index], new_schedule['previous_solution_id'].iloc[index]))
    for pair in prev_flights:
        if pair not in new_flights:
            diff += 1
    # print(f'Schedule difference is {diff}')
    return diff


def ts_time_shifts(new_schedule: pd.DataFrame,
                   technical_service_table: pd.DataFrame,
                   aircraft_equipment_table: pd.DataFrame,
                   current_time: datetime) -> int:
    """Суммарное количество сдвинутых ТО -> min"""
    technical_service_aircraft_ids = technical_service_table['aircraft_id'].unique().tolist()
    summary_count = 0
    for aircraft_id in technical_service_aircraft_ids:
        aircraft_info = \
            aircraft_equipment_table[aircraft_equipment_table['aircraft_id'] == aircraft_id]['equipment_id'].iloc[0]
        if aircraft_info == 0:
            continue
        aircraft_flights = aircraft_flight_line(aircraft_id, new_schedule)
        aircraft_ts = technical_service_table[technical_service_table['aircraft_id'] == aircraft_id]
        technical_service_ids = aircraft_ts['technical_service_id'].unique().tolist()
        ts_used_time = []
        ts_allowed_time = []
        for technical_service_id in technical_service_ids:
            time_size = aircraft_ts[aircraft_ts['technical_service_id'] == technical_service_id]['time_size'].iloc[0]
            time_size = pd.to_timedelta(time_size)
            if not allowed_time_differences(aircraft_flights, time_size, ts_allowed_time, current_time):
                # Штраф за невозможность проведения ТО
                print(f'TS trouble with aircraft {aircraft_id} (can not do ts)')
                summary_count += 1000
            else:
                count = checking_allowed_ts_time(technical_service_table, ts_allowed_time, ts_used_time)
                if count >= 1000:
                    print(f'TS trouble with aircraft {aircraft_id}')
                summary_count += count
    # print(f'Count of moved TS is {summary_count}')
    return summary_count


def schedule_time_checker(new_schedule: pd.DataFrame, aircraft_equipment_table: pd.DataFrame) -> bool:
    """Проверяет, не накладываются ли по времени рейсы в новом расписании(min время между рейсами 30 минут)"""
    aircraft_ids = new_schedule['aircraft_id'].unique().tolist()
    flag = True
    for aircraft_id in aircraft_ids:
        aircraft_info = \
            aircraft_equipment_table[aircraft_equipment_table['aircraft_id'] == aircraft_id]['equipment_id'].iloc[0]
        if aircraft_info == 0:
            continue
        aircraft_flights = aircraft_flight_line(aircraft_id, new_schedule)
        for index in range(len(aircraft_flights) - 1):
            arrival_time = aircraft_flights['arrival_time'].iloc[index]
            next_departure_time = aircraft_flights['departure_time'].iloc[index + 1]
            if (pd.to_datetime(next_departure_time) - pd.to_datetime(arrival_time)).total_seconds() < 30 * 60:
                flag = False
                # # print(f'Bad timedelta is {(pd.to_datetime(next_departure_time) - pd.to_datetime(arrival_time)).total_seconds()}')
                print(
                    f'Now checking {pd.to_datetime(arrival_time)} and {pd.to_datetime(next_departure_time)} for aircraft {aircraft_id}')
            # print(f'Time checker is {flag}')
    # print(f'Schedule time checker is {flag}')
    return flag


def airports_checker(new_schedule: pd.DataFrame, aircraft_equipment_table: pd.DataFrame) -> bool:
    """Проверяет не накладываются ли рейсы по нахождению в разных аэропортах"""
    aircraft_ids = new_schedule['aircraft_id'].unique().tolist()
    flag = True
    for aircraft_id in aircraft_ids:
        aircraft_info = \
            aircraft_equipment_table[aircraft_equipment_table['aircraft_id'] == aircraft_id]['equipment_id'].iloc[0]
        if aircraft_info == 0:
            continue
        aircraft_flight = aircraft_flight_line(aircraft_id, new_schedule)
        # if aircraft_id >= 14:
        #     print(f'For aircraft {aircraft_id} flights are {aircraft_flight}')
        for index in range(len(aircraft_flight) - 1):
            arrival_airport = aircraft_flight['arrival_airport_code'].iloc[index]
            next_departure_airport = aircraft_flight['departure_airport_code'].iloc[index + 1]
            if arrival_airport != next_departure_airport:
                flag = False
                print(f'Airports checker {aircraft_id} is {flag}')
    return flag


def equipment_checker(new_schedule: pd.DataFrame,
                      flight_equipment_table: pd.DataFrame,
                      aircraft_equipment_table: pd.DataFrame) -> bool:
    """Проверяет, подходит ли оснащение самолёта для его рейсов"""
    flag = True
    aircraft_ids = new_schedule['aircraft_id'].unique().tolist()
    for aircraft_id in aircraft_ids:
        aircraft_equipment = \
            aircraft_equipment_table[aircraft_equipment_table['aircraft_id'] == aircraft_id]['equipment_id'].iloc[0]
        aircraft_flights = aircraft_flight_line(aircraft_id, new_schedule)
        # print(aircraft_flights)
        for index, flight_id in aircraft_flights['flight_id'].items():
            # print(flight_id)
            flight_equipment = \
                flight_equipment_table[flight_equipment_table['flight_id'] == flight_id]['equipment_ids'].iloc[0]
            flight_equipment = list(ast.literal_eval(flight_equipment))
            if aircraft_equipment not in flight_equipment:
                flag = False
                print(
                    f'Now checking aircraft {aircraft_id} with {aircraft_equipment} and flight {flight_id} with eqipment {flight_equipment}')
        # print(f'Equipment checker is {flag}')

    return flag


def penalty_function(new_schedule: pd.DataFrame,
                     schedule_for_equipment_checker: pd.DataFrame,
                     flight_equipment_table: pd.DataFrame,
                     aircraft_equipment_table: pd.DataFrame) -> int:
    penalty = 1000
    a = schedule_time_checker(new_schedule, aircraft_equipment_table)
    if not a:
        print(f'Time checker false')
    b = airports_checker(new_schedule, aircraft_equipment_table)
    if not b:
        print(f'Airports checker is false')
    c = equipment_checker(schedule_for_equipment_checker, flight_equipment_table, aircraft_equipment_table)
    if not c:
        print(f'Equipment checker is false')
    # if not (schedule_time_checker(new_schedule)
    #         and airports_checker(new_schedule)
    #         and equipment_checker(schedule_for_equipment_checker, flight_equipment_table, aircraft_equipment_table)):
    #     return penalty
    if not (a and b and c):
        return penalty
    # print(f'All checkers is True')
    return 0


def objective_function(previous_schedule: pd.DataFrame,
                       new_schedule: pd.DataFrame,
                       schedule_part_for_penalty: pd.DataFrame,
                       schedule_part_for_equipment_penalty: pd.DataFrame,
                       flight_equipment_table: pd.DataFrame,
                       aircraft_equipment_table: pd.DataFrame,
                       technical_service_table: pd.DataFrame,
                       current_time: datetime) -> int:
    return (schedule_differences(previous_schedule, new_schedule)
            + ts_time_shifts(new_schedule, technical_service_table, aircraft_equipment_table, current_time)
            + penalty_function(schedule_part_for_penalty, schedule_part_for_equipment_penalty, flight_equipment_table,
                               aircraft_equipment_table))


# def list_based_sa_algorithm(temperature_list: list,
#                             max_iteration_times: int,
#                             markov_chain_length: int,
#                             equipment_disruption_table: pd.DataFrame,
#                             time_shift_disruption_table: pd.DataFrame,
#                             previous_schedule: pd.DataFrame,
#                             flight_equipments_table: pd.DataFrame,
#                             aircraft_table: pd.DataFrame,
#                             technical_service_table,
#                             current_time: datetime) -> int:
#     outer_loop_iterator = 0
#
#     # В текущем расписании обновили время взлета и посадки рейсов в соответствии с disruption-таблицей переноса рейсов
#     updated_previous_solution = change_disrupted_flights_time(time_shift_disruption_table, previous_schedule, 0, 1, 2,
#                                                               3)
#     # В измененном текущем расписании выбрали ближайшие 3 дня
#     nearest_flights = nearest_flights_selection(updated_previous_solution, current_time, 'KJA', 'LED')
#     nearest_flights.to_csv('csv_files/new_schedule_NEARST.csv', index=False, sep=';')
#     # Извлекаем из таблицы тех обслуживания строки, попадающие в нужный диапазон времени
#     technical_service_table = remake_technical_service_table(technical_service_table, nearest_flights)
#     # Текущее решение = ближайшие 3 дня из расписания
#     current_solution = nearest_flights
#
#     # Вычисляем objective function для текущего расписания
#     current_solution_objective_func = objective_function(current_solution,
#                                                          current_solution,
#                                                          current_solution,
#                                                          current_solution,
#                                                          flight_equipments_table,
#                                                          aircraft_table,
#                                                          technical_service_table)
#     print(f'Begin with objective function = {current_solution_objective_func}')
#     # result_list = []
#     probability_list = []
#     exponent_list = []
#     # temperature_list_for_plot = []
#     # objective_functions_list = []
#     start = time.time()
#
#     while outer_loop_iterator <= max_iteration_times:
#         temperature_max = max(temperature_list)
#         # temperature_list_for_plot.append(temperature_max)
#         outer_loop_iterator += 1
#         temperature = 0
#         bad_solution_count, inner_loop_iterator = 0, 0
#         while inner_loop_iterator <= markov_chain_length:
#             # Случайно выбрать 2 ВС и случайно выбрать flight_id для триггерного ВС,
#             # сделать новое расписание через swap
#
#             # Создали словарь, где ключи - ВС с триггером, значения - рейсы которые не могут быть выполнены
#             trigger_aircraft_and_flight_dict = all_trigger_aircraft_and_flight(equipment_disruption_table,
#                                                                                time_shift_disruption_table,
#                                                                                flight_equipments_table,
#                                                                                nearest_flights,
#                                                                                current_time)
#
#             # Сделали список со всеми триггерными ВС
#             # trigger_aircraft_ids = list(trigger_aircraft_and_flight_dict.keys())
#
#             # Случайно выбрали триггерное ВС и рейс для него
#             trigger_aircraft, trigger_flight_id = aircraft_and_flight_random_choice(trigger_aircraft_and_flight_dict)
#             # Случайно выбрали второе ВС для перестановки
#             aircraft_to_swap = aircraft_for_swap_random_choice(aircraft_table, trigger_aircraft)
#             # Сделали разбиение текущего расписание на связки
#             current_schedule_partition = base_airports_partition(current_solution, 'KJA', 'LED')
#
#             print(
#                 f'Now iteration with trigger aircraft {trigger_aircraft}, flight {trigger_flight_id}, second aircraft {aircraft_to_swap}')
#
#             current_flights_for_penalty = current_solution.loc[
#                 (current_solution['aircraft_id'] == trigger_aircraft)
#                 | (current_solution['aircraft_id'] == aircraft_to_swap)]
#
#             current_flights_for_equipment_penalty = current_solution.loc[
#                 ((current_solution['aircraft_id'] == trigger_aircraft)
#                  & (current_solution['flight_id'] == trigger_flight_id['flight_id'])
#                  & (current_solution['previous_solution_id'] == trigger_flight_id['previous_solution_id']))]
#
#             objective_function_before_swap = objective_function(nearest_flights,
#                                                                 current_solution,
#                                                                 current_flights_for_penalty,
#                                                                 current_flights_for_equipment_penalty,
#                                                                 flight_equipments_table,
#                                                                 aircraft_table,
#                                                                 technical_service_table)
#
#             print(f'Objective function before swap for 2 aircrafts {objective_function_before_swap}')
#
#             swapped_flag = [False]
#             # С помощью smart_swap сделали candidate_solution
#             candidate_solution = smart_swap(trigger_aircraft,
#                                             aircraft_to_swap,
#                                             current_schedule_partition,
#                                             nearest_flights,
#                                             trigger_flight_id,
#                                             flight_equipments_table,
#                                             aircraft_table,
#                                             swapped_flag)
#
#             inner_loop_iterator += 1
#
#             candidate_flights_for_penalty_swapped = candidate_solution.loc[
#                 ((candidate_solution['aircraft_id'] == trigger_aircraft)
#                  | (candidate_solution['aircraft_id'] == aircraft_to_swap))]
#
#             if swapped_flag[0]:
#                 swapped_aircraft = aircraft_to_swap
#             else:
#                 swapped_aircraft = trigger_aircraft
#
#             candidate_flights_for_equipment_penalty_swapped = candidate_solution.loc[
#                 ((candidate_solution['aircraft_id'] == swapped_aircraft)
#                  & (candidate_solution['previous_solution_id'] == trigger_flight_id['previous_solution_id']))]
#
#             objective_function_after_swap = objective_function(nearest_flights,
#                                                                candidate_solution,
#                                                                candidate_flights_for_penalty_swapped,
#                                                                candidate_flights_for_equipment_penalty_swapped,
#                                                                flight_equipments_table,
#                                                                aircraft_table,
#                                                                technical_service_table)
#             print(f'Objective function after swap for 2 aircrafts {objective_function_after_swap}')
#
#             # Вычисляем objective function для нового расписания(сделали smart_swap)
#             candidate_solution_objective_func = (current_solution_objective_func
#                                                  - objective_function_before_swap
#                                                  + objective_function_after_swap)
#
#             print(
#                 f'Now iteration {outer_loop_iterator}, {inner_loop_iterator}. Chosen trigger aircraft: {trigger_aircraft}, trigger flight: {trigger_flight_id}, second aircraft: {aircraft_to_swap}, current OF {current_solution_objective_func}, candidate OF {candidate_solution_objective_func}')
#
#             if 0 < candidate_solution_objective_func < current_solution_objective_func:
#                 current_solution = candidate_solution
#                 # Вычисляем objective function для текущего расписания
#                 current_solution_objective_func = objective_function(nearest_flights,
#                                                                      current_solution,
#                                                                      current_solution,
#                                                                      current_solution,
#                                                                      flight_equipments_table,
#                                                                      aircraft_table,
#                                                                      technical_service_table)
#                 print(f'Found better solution {current_solution_objective_func}')
#                 current_solution.to_csv('csv_files/new_schedule_RESULT_' + str(outer_loop_iterator)
#                                         + '_' + str(inner_loop_iterator) + '.csv', index=False, sep=';')
#                 probability = 1
#             else:
#                 safe_temperature_max = max(temperature_max, 1e-10)
#                 exponent = -(candidate_solution_objective_func - current_solution_objective_func) / safe_temperature_max
#                 exponent_list.append(exponent)
#                 probability = math.exp(exponent)
#                 random_float = random.random()
#                 if random_float < probability:
#                     # когда temperature == 0 и candidate_solution_distance_sum == current_solution_distance_sum получается число меньшее 1e-10
#                     temperature = max(
#                         (temperature - candidate_solution_objective_func + current_solution_objective_func) / math.log(
#                             random_float), 1e-10)
#                     # temperature_list_for_plot.append(temperature_max)
#
#                     # if temperature == 1e-10:
#                     #     print(prev_temp, candidate_solution_distance_sum - current_solution_distance_sum, math.log(random_float))
#
#                     bad_solution_count += 1
#                     current_solution = candidate_solution
#                     # Вычисляем objective function для текущего расписания
#                     current_solution_objective_func = objective_function(nearest_flights,
#                                                                          current_solution,
#                                                                          current_solution,
#                                                                          current_solution,
#                                                                          flight_equipments_table,
#                                                                          aircraft_table,
#                                                                          technical_service_table)
#                     print(f'Randomly found better solution {current_solution_objective_func}')
#                     # current_solution.to_csv('csv_files/new_schedule_RANDOM_RESULT_' + str(outer_loop_iterator)
#                     #                         + '_' + str(inner_loop_iterator) + '.csv', index=False, sep=';')
#                 else:
#                     pass
#             # objective_functions_list.append(current_solution_distance_sum)
#             probability_list.append(probability)
#             # result_list.append(calculate_distance_of_permutation(current_solution))
#
#         if bad_solution_count != 0:
#             temperature_list.remove(temperature_max)
#             # temperature_list_for_plot.append(temperature_max)
#             temperature_list.append(temperature / bad_solution_count)
#
#     # print(probability_list)
#     # plt.plot(range(5000), temperature_list_for_plot[:5000])
#     # plt.plot(range(5000), objective_functions_list[:5000])
#     plt.plot(range(max_iteration_times), probability_list[:max_iteration_times])
#     # plt.plot(range(max_iteration_times*markov_chain_length), result_list[:max_iteration_times*markov_chain_length])
#     # plt.plot(range(30), exponent_list[:30])
#     plt.plot()
#     plt.show()
#     print(time.time() - start)
#     current_solution.to_csv('csv_files/new_schedule_RESULT.csv', index=False, sep=';')
#     return current_solution_objective_func


def local_optimisation_algorithm(equipment_disruption_table: pd.DataFrame,
                                 time_shift_disruption_table: pd.DataFrame,
                                 previous_schedule: pd.DataFrame,
                                 flight_equipments_table: pd.DataFrame,
                                 aircraft_table: pd.DataFrame,
                                 technical_service_table,
                                 current_time: datetime) -> tuple:
    # В текущем расписании обновили время взлета и посадки рейсов в соответствии с disruption-таблицей переноса рейсов
    updated_previous_solution = change_disrupted_flights_time(time_shift_disruption_table, previous_schedule,
                                                              ['KJA', 'LED'], 0, 1, 2, 3)
    # В измененном текущем расписании выбрали ближайшие 3 дня
    nearest_flights = nearest_flights_selection(updated_previous_solution, current_time, 'KJA', 'LED')
    nearest_flights.to_csv('csv_files/new_schedule_NEARST.csv', index=False, sep=';')
    first_partition = base_airports_partition(nearest_flights, 'KJA', 'LED')
    # Извлекаем из таблицы тех обслуживания строки, попадающие в нужный диапазон времени
    technical_service_table = remake_technical_service_table(technical_service_table, nearest_flights)
    # Текущее решение = ближайшие 3 дня из расписания
    current_solution = nearest_flights

    # Вычисляем objective function для текущего расписания
    current_solution_objective_func = objective_function(nearest_flights,
                                                         current_solution,
                                                         current_solution,
                                                         current_solution,
                                                         flight_equipments_table,
                                                         aircraft_table,
                                                         technical_service_table,
                                                         current_time)
    print(f'Begin with objective function = {current_solution_objective_func}')

    # Создали словарь, где ключи - ВС с триггером, значения - рейсы которые не могут быть выполнены
    trigger_aircraft_and_flight_dict = all_trigger_aircraft_and_flight(equipment_disruption_table,
                                                                       time_shift_disruption_table,
                                                                       flight_equipments_table,
                                                                       nearest_flights,
                                                                       current_time)
    # Сделали список со всеми триггерными ВС
    trigger_aircraft_ids = list(trigger_aircraft_and_flight_dict.keys())

    list_for_flights_check = list(itertools.chain.from_iterable(list(trigger_aircraft_and_flight_dict.values())))
    print(list_for_flights_check)

    # Сделали список из всех доступных ВС(у кого оборудование не 0)
    all_aircraft_ids = aircraft_table['aircraft_id'].unique().tolist()
    for aircraft_id in all_aircraft_ids:
        aircraft_info = aircraft_table[aircraft_table['aircraft_id'] == aircraft_id]['equipment_id'].iloc[0]
        if aircraft_info == 0:
            all_aircraft_ids.remove(aircraft_id)
    print(all_aircraft_ids)
    count = 0

    dict_of_all_swapped_flights = {'was_triggered': [], 'was_health': []}

    start = time.time()

    for trigger_aircraft in trigger_aircraft_ids:
        for flight_id in trigger_aircraft_and_flight_dict[trigger_aircraft]:
            # Сделали разбиение текущего расписание на связки
            current_schedule_partition = base_airports_partition(current_solution, 'KJA', 'LED')
            # Список всех рейсов триггерного ВС
            current_trigger_aircraft_flights = current_solution[current_solution['aircraft_id'] == trigger_aircraft][
                'previous_solution_id'].tolist()
            # Инициализируем переменную для сохранения временного результата
            temp_current = current_solution
            temp_current_solution_objective_func = current_solution_objective_func
            temp_dict_of_swapped_flights = {'was_triggered': [], 'was_health': []}
            for aircraft_to_swap in all_aircraft_ids:
                if flight_id['previous_solution_id'] in current_trigger_aircraft_flights:
                    if aircraft_to_swap != trigger_aircraft:
                        # print(
                        #     f'Now iteration with trigger aircraft {trigger_aircraft}, flight_id {flight_id}, second aircraft {aircraft_to_swap}')

                        current_flights_for_penalty = current_solution.loc[
                            (current_solution['aircraft_id'] == trigger_aircraft)
                            | (current_solution['aircraft_id'] == aircraft_to_swap)]

                        current_flights_for_equipment_penalty = current_solution.loc[
                            ((current_solution['aircraft_id'] == trigger_aircraft)
                             & (current_solution['flight_id'] == flight_id['flight_id'])
                             & (current_solution['previous_solution_id'] == flight_id['previous_solution_id']))]

                        objective_function_before_swap = objective_function(nearest_flights,
                                                                            current_solution,
                                                                            current_flights_for_penalty,
                                                                            current_flights_for_equipment_penalty,
                                                                            flight_equipments_table,
                                                                            aircraft_table,
                                                                            technical_service_table,
                                                                            current_time)

                        print(f'Objective function before swap for 2 aircrafts {objective_function_before_swap}')

                        swapped_flag = [False]

                        # С помощью smart_swap сделали candidate_solution
                        candidate_solution, candidate_dict_of_swapped_flag = smart_swap(trigger_aircraft,
                                                                                        aircraft_to_swap,
                                                                                        current_schedule_partition,
                                                                                        nearest_flights,
                                                                                        flight_id,
                                                                                        flight_equipments_table,
                                                                                        aircraft_table,
                                                                                        swapped_flag)

                        candidate_flights_for_penalty_swapped = candidate_solution.loc[
                            ((candidate_solution['aircraft_id'] == trigger_aircraft)
                             | (candidate_solution['aircraft_id'] == aircraft_to_swap))]

                        if swapped_flag[0]:
                            swapped_aircraft = aircraft_to_swap
                        else:
                            swapped_aircraft = trigger_aircraft

                        candidate_flights_for_equipment_penalty_swapped = candidate_solution.loc[
                            ((candidate_solution['aircraft_id'] == swapped_aircraft)
                             & (candidate_solution['previous_solution_id'] == flight_id['previous_solution_id']))]

                        objective_function_after_swap = objective_function(nearest_flights,
                                                                           candidate_solution,
                                                                           candidate_flights_for_penalty_swapped,
                                                                           candidate_flights_for_equipment_penalty_swapped,
                                                                           flight_equipments_table,
                                                                           aircraft_table,
                                                                           technical_service_table,
                                                                           current_time)

                        print(f'Objective function after swap for 2 aircrafts {objective_function_after_swap}')

                        # Вычисляем objective function для нового расписания(сделали smart_swap)
                        candidate_solution_objective_func = (current_solution_objective_func
                                                             - objective_function_before_swap
                                                             + objective_function_after_swap)

                        print(
                            f'Now iteration with trigger aircraft {trigger_aircraft}, flight_id {flight_id}, second aircraft {aircraft_to_swap}, current OF {temp_current_solution_objective_func}, candidate OF {candidate_solution_objective_func}')

                        if 0 < candidate_solution_objective_func < temp_current_solution_objective_func:
                            # Сохраняем текущее лучшее решение
                            temp_current_solution_objective_func = candidate_solution_objective_func
                            temp_current = candidate_solution
                            temp_dict_of_swapped_flights = candidate_dict_of_swapped_flag
                            print(
                                f'Found better schedule with objective function = {temp_current_solution_objective_func}. Save to file {count}')
                            temp_current.to_csv('csv_files/new_schedule_RESULT_' + str(count) + '.csv', index=False,
                                                sep=';')
                            count += 1
            if flight_id['previous_solution_id'] in current_trigger_aircraft_flights:
                list_for_flights_check.remove(flight_id)
            current_solution = temp_current
            current_solution_objective_func = objective_function(nearest_flights,
                                                                 current_solution,
                                                                 current_solution,
                                                                 current_solution,
                                                                 flight_equipments_table,
                                                                 aircraft_table,
                                                                 technical_service_table,
                                                                 current_time)
            dict_of_all_swapped_flights['was_triggered'].extend(temp_dict_of_swapped_flights['was_triggered'])
            dict_of_all_swapped_flights['was_health'].extend(temp_dict_of_swapped_flights['was_health'])
    current_solution.to_csv('csv_files/new_schedule_RESULT.csv', index=False, sep=';')
    print(list_for_flights_check)
    print(time.time() - start)
    dict_of_all_swapped_flights = {
        'was_triggered': [d['previous_solution_id'] for sublist in dict_of_all_swapped_flights.get('was_triggered', []) for d in sublist if
                          'previous_solution_id' in d],
        'was_health': [d['previous_solution_id'] for sublist in dict_of_all_swapped_flights.get('was_health', []) for d in sublist if
                       'previous_solution_id' in d]
    }
    print(f'Переставленные рейсы {dict_of_all_swapped_flights}')
    return current_solution_objective_func, dict_of_all_swapped_flights


"""Создаем необходимые данные для отжига"""
# Список температур
temperature_list = [1000, 500, 250, 125, 60, 30, 15, 7, 3, 1]
# Максимальное количество внешних итераций
max_outer_iteration_times = 5
# Максимальное количество внутренних итераций
max_inner_iteration_times = 5
# DataFrame со списком disruptions по смене оборудования ВС
table_with_equipment_disruptions = problematic_aircraft_equipment
# DataFrame со списком disruptions по переносу времени рейсов
table_with_flight_shift_disruptions = problematic_flight_shift
# Предыдущее решение, которое было до disruptions
previous_schedule = previous_solution
# Таблица с необходимым оборудованием для каждого рейса
table_with_flight_equipments_info = flight_equipments
# Таблица с данными по оборудованию каждого ВС
table_with_aircraft_equipment_info = updated_aircraft
# Таблица с данными по ТО
table_with_technical_service_info = technical_service
# Текущий момент времени
test_1_curr_time = datetime(2025, 1, 22, 0, 0)
test_2_curr_time = datetime(2025, 1, 21, 0, 0)

result_schedule, dict_of_swapped = local_optimisation_algorithm(table_with_equipment_disruptions,
                                                                table_with_flight_shift_disruptions,
                                                                previous_schedule,
                                                                table_with_flight_equipments_info,
                                                                table_with_aircraft_equipment_info,
                                                                table_with_technical_service_info,
                                                                test_1_curr_time)
# dict_of_swapped = {'was_triggered': [10, 11, 54, 55, 57, 58, 59, 60, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 205, 206], 'was_health': []}

# print(list_based_sa_algorithm(temperature_list,
#                               max_outer_iteration_times,
#                               max_inner_iteration_times,
#                               table_with_equipment_disruptions,
#                               table_with_flight_shift_disruptions,
#                               previous_schedule,
#                               table_with_flight_equipments_info,
#                               table_with_aircraft_equipment_info,
#                               table_with_technical_service_info,
#                               curr_time))

nearest = nearest_flights_selection(previous_schedule, test_1_curr_time, 'KJA', 'LED')
gantt_chart(nearest, dict_of_swapped)

result = pd.read_csv('csv_files/new_schedule_RESULT.csv', sep=';')
gantt_chart(result, dict_of_swapped)

ts = remake_technical_service_table(table_with_technical_service_info, nearest)
part = base_airports_partition(nearest, 'KJA', 'LED')
f = [False]
# after_swap = smart_swap(3, 16, part, nearest, {'flight_id': 'FV6877', 'aircraft_id': 3, 'previous_solution_id': 59}, flight_equipments, updated_aircraft, f)
# after_swap.to_csv('csv_files/test.csv', index=False, sep=';')
# current_flights_for_penalty = after_swap.loc[
#                             (after_swap['aircraft_id'] == 3)
#                             | (after_swap['aircraft_id'] == 16)]
#
# current_flights_for_equipment_penalty = after_swap.loc[
#     ((after_swap['aircraft_id'] == 16)
#      & (after_swap['flight_id'] == 'FV6877')
#      & (after_swap['previous_solution_id'] == 59))]
# print(objective_function(nearest, after_swap, current_flights_for_penalty, current_flights_for_equipment_penalty, table_with_flight_equipments_info, table_with_aircraft_equipment_info, ts))


# t = pd.read_csv('csv_files/t1.csv', sep=';')
# t_part = base_airports_partition(t, 'KJA', 'LED')
# print(t_part)
#
# print(from_partition_to_dataframe(t_part))
#
# from_partition_to_dataframe(t_part).to_csv('csv_files/t2.csv', index=False, sep=';')


print(objective_function(nearest, nearest, nearest, nearest, table_with_flight_equipments_info,
                         table_with_aircraft_equipment_info, ts, test_1_curr_time))

print(objective_function(nearest, result, result, result, table_with_flight_equipments_info,
                         table_with_aircraft_equipment_info, ts, test_1_curr_time))

# print(f)

# #
# after_swap = smart_swap(9, 18, part, nearest, {'flight_id': 'FV6342', 'aircraft_id': 9, 'previous_solution_id': 206}, flight_equipments, updated_aircraft)
# after_swap.to_csv('csv_files/test.csv', index=False, sep=';')
# test = from_partition_to_dataframe(part)
# test.to_csv('csv_files/test.csv', index=False, sep=";")


# updated_previous_solution = change_disrupted_flights_time(table_with_flight_shift_disruptions, previous_schedule, 0, 1, 2, 3)
# nearest_flights = nearest_flights_selection(updated_previous_solution, curr_time, 'KJA', 'LED')
# parts = base_airports_partition(updated_previous_solution, 'KJA', 'LED')

# print(smart_swap(12, 11, parts, nearest_flights, [3, 12], {'flight_id': 'FV6234', 'aircraft_id': 12, 'previous_solution_id': 295}, table_with_flight_equipments_info, table_with_aircraft_equipment_info))
