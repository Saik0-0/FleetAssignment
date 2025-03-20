import ast
from itertools import chain
import pandas as pd
from ast import literal_eval
from datetime import datetime, timedelta
from dateutil import parser


def change_aircraft_equipment(problematic_aircraft_equipment_table: pd.DataFrame,
                              aircraft_table: pd.DataFrame,
                              *disruption_ids: int) -> pd.DataFrame:
    """Меняет данные по оснащению самолетов в соответствии с таблицей сбоев"""
    problematic_aircraft_equipment_table = problematic_aircraft_equipment_table.loc[[*disruption_ids]]
    problematic_aircrafts = problematic_aircraft_equipment_table['aircraft_id'].tolist()
    new_equipment = {}
    for aircraft in problematic_aircrafts:
        new_equipment[aircraft] = problematic_aircraft_equipment_table[problematic_aircraft_equipment_table['aircraft_id'] == aircraft]['equipment_id'].iloc[0]
    for ind in aircraft_table.index:
        if aircraft_table['aircraft_id'].iloc[ind] in problematic_aircrafts:
            aircraft_table.loc[ind, 'equipment_id'] = new_equipment[aircraft_table['aircraft_id'].iloc[ind]]
    return aircraft_table


def change_disrupted_flights_time(problematic_flight_shift_table: pd.DataFrame,
                                  previous_schedule: pd.DataFrame,
                                  *disruption_ids: int) -> pd.DataFrame:
    """Меняет данные о времени вылета и прилета рейсов в соответствии с таблицей сбоев"""
    problematic_flight_shift_table = problematic_flight_shift_table.loc[[*disruption_ids]]
    problematic_ids = problematic_flight_shift_table['previous_solution_id'].tolist()
    time_shifts = {}
    for problematic_id in problematic_ids:
        time_shifts[problematic_id] = problematic_flight_shift_table[problematic_flight_shift_table['previous_solution_id'] == problematic_id]['shift'].iloc[0]
        time_shifts[problematic_id] = pd.to_timedelta(time_shifts[problematic_id])
    for ind in previous_schedule.index:
        if previous_schedule['previous_solution_id'].iloc[ind] in problematic_ids:
            previous_schedule.loc[ind, 'departure_time'] = pd.to_datetime(previous_schedule.loc[ind, 'departure_time'], dayfirst=True) + time_shifts[previous_schedule['previous_solution_id'].iloc[ind]]
            previous_schedule.loc[ind, 'arrival_time'] = pd.to_datetime(previous_schedule.loc[ind, 'arrival_time'], dayfirst=True) + time_shifts[previous_schedule['previous_solution_id'].iloc[ind]]
    return previous_schedule


pd.set_option("display.max_columns", None)
# парк доступных вс
aircraft = pd.read_csv('csv_files/df_aircraft.csv', sep=';')
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
aircraft = change_aircraft_equipment(problematic_aircraft_equipment, aircraft, 0, 1, 2, 3, 4)
# aircraft.to_csv('csv_files/new_aircraft.csv', index=False, sep=';')

previous_solution = change_disrupted_flights_time(problematic_flight_shift, previous_solution, 0, 1, 2, 3)
# previous_solution.to_csv('csv_files/new_previous_solution.csv', index=False, sep=';')


def nearest_flights_selection(previous_solution_table: pd.DataFrame,
                              current_time: datetime,
                              *base_airports: str) -> pd.DataFrame:
    """Возвращает часть расписания (указанная дата + 3 суток)"""
    previous_solution_table_result = pd.DataFrame(columns=previous_solution_table.columns)
    for index, flight_row in previous_solution_table.iterrows():
        if current_time <= pd.to_datetime(flight_row['departure_time'], dayfirst=True) <= current_time + timedelta(
                days=2, hours=23, minutes=59):
            if (index + 1 < len(previous_solution_table.index) and pd.to_datetime(previous_solution_table.iloc[index + 1]['departure_time'],
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
        new_equipment_dict[aircraft_id] = \
            disruptions_table[disruptions_table['aircraft_id'] == aircraft_id]['equipment_id'].tolist()[0]

    problematic_flights = previous_solution_table[previous_solution_table['aircraft_id'].isin(problematic_aircrafts)]

    # Структура словаря: ключи - id самолетов с изменением оборудования,
    # значения - set из id рейсов куда распределены самолеты
    problematic_flights_id = {}
    for aircraft_id in problematic_aircrafts:
        problematic_flights_id[aircraft_id] = set(
            problematic_flights[problematic_flights['aircraft_id'] == aircraft_id]['flight_id'].tolist())

    disrupted_flights = []
    for aircraft_id in problematic_aircrafts:
        flight_equipments_subtable = flight_equipments_table[
            flight_equipments_table['flight_id'].isin(problematic_flights_id[aircraft_id])]
        for flight_id in problematic_flights_id[aircraft_id]:
            previous_solution_id = \
                previous_solution_table[previous_solution_table['flight_id'] == flight_id]['previous_solution_id'].iloc[
                    0]
            if new_equipment_dict[aircraft_id] not in literal_eval(
                    flight_equipments_subtable[flight_equipments_subtable['flight_id'] == flight_id][
                        'equipment_ids'].iloc[0]):
                disrupted_flights.append({'flight_id': flight_id, 'aircraft_id': aircraft_id,
                                          'previous_solution_id': int(previous_solution_id)})

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
    # print(string_time, type(string_time))
    if type(string_time) == str:
        time_object = parser.parse(string_time, dayfirst=True)
    else:
        return string_time
    return time_object


def flight_shift_disrupted_flights(previous_solution_table: pd.DataFrame,
                                   disruptions_table: pd.DataFrame,
                                   *disruption_ids: int) -> list:
    """Возвращает список полётов, которые становятся невыполнимыми из-за переноса рейсов
    (в т.ч. проверка на +50 минут после рейса"""
    problematic_aircrafts = disruptions_table['aircraft_id'].tolist()
    disrupted_flights = []

    for disruption_id in disruption_ids:
        previous_solution_id = disruptions_table[disruptions_table['problematic_flight_shift_id'] == disruption_id][
            'previous_solution_id'].iloc[0]
        if previous_solution_id not in previous_solution_table['previous_solution_id'].values:
            continue

        new_arrival_time = get_time_from_table(disruptions_table, previous_solution_id, 'new_arrival_time')
        next_departure_time = get_time_from_table(previous_solution_table, previous_solution_id, 'departure_time')

        flight_id = previous_solution_table[previous_solution_table['previous_solution_id'] == previous_solution_id][
            'flight_id'].iloc[0]
        aircraft_id = problematic_aircrafts[disruption_id]

        time_delta = next_departure_time - new_arrival_time
        # 30 минут = минимальное окно между рейсами
        if time_delta.total_seconds() < 30 * 60:
            disrupted_flights.append(
                {'flight_id': flight_id, 'aircraft_id': aircraft_id, 'previous_solution_id': int(previous_solution_id)})

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
    departure_time_row = pd.to_datetime(previous_solution_table['departure_time'], dayfirst=True)
    arrival_time_row = pd.to_datetime(previous_solution_table['arrival_time'], dayfirst=True)
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


def aircraft_flight_line(aircraft_id: int, nearest_schedule: pd.DataFrame) -> pd.DataFrame:
    """Возвращает DataFrame полетов для конкретного ВС"""
    aircraft_flights = nearest_schedule[nearest_schedule['aircraft_id'] == aircraft_id].reset_index(drop=True)
    return aircraft_flights


def extract_part_using_flight_id(aircraft_id: int, flight_id: str, airports_partition: list) -> list:
    """Возвращает связку с конкретным рейсом flight_id"""
    trigger_part = []
    for part in airports_partition:
        for flight in part:
            if flight_id == flight['flight_id'] and aircraft_id == flight['aircraft_id']:
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
            return extract_part_using_flight_id(aircraft_id, flight['flight_id'], airport_partition)


def extract_trigger_time_range(aircraft_id: int, trigger_flight_id: str, airports_partition: list) -> list:
    """Возвращает временной отрезок, в котором находится связка с триггерным рейсом"""
    trigger_part = extract_part_using_flight_id(aircraft_id, trigger_flight_id, airports_partition)
    trigger_departure_time = trigger_part[0]['departure_time']
    trigger_arrival_time = trigger_part[-1]['arrival_time']
    return [trigger_departure_time, trigger_arrival_time]


def from_partition_to_dataframe(partition_list: list) -> pd.DataFrame:
    """Из списка связок получаем расписание в DataFrame"""
    temp_list = list(chain(*partition_list))
    new_schedule = pd.DataFrame(temp_list)
    new_schedule['departure_time'] = new_schedule['departure_time'].astype(str)
    new_schedule['arrival_time'] = new_schedule['arrival_time'].astype(str)
    new_schedule = new_schedule.sort_values(by=['aircraft_id', 'departure_time']).reset_index(drop=True)

    return new_schedule


def schedule_time_checker(new_schedule: pd.DataFrame) -> bool:
    """Проверяет, не накладываются ли по времени рейсы в новом расписании(min время между рейсами 30 минут)"""
    aircraft_ids = new_schedule['aircraft_id'].unique().tolist()
    flag = True
    for aircraft_id in aircraft_ids:
        aircraft_flights = aircraft_flight_line(aircraft_id, new_schedule)
        for index in range(len(aircraft_flights) - 1):
            arrival_time = aircraft_flights['arrival_time'].iloc[index]
            next_departure_time = aircraft_flights['departure_time'].iloc[index + 1]
            if (pd.to_datetime(next_departure_time) - pd.to_datetime(arrival_time)).total_seconds() < 30 * 60:
                flag = False
    return flag


def airports_checker(new_schedule: pd.DataFrame) -> bool:
    """Проверяет не накладываются ли рейсы по нахождению в разных аэропортах"""
    aircraft_ids = new_schedule['aircraft_id'].unique().tolist()
    flag = True
    for aircraft_id in aircraft_ids:
        aircraft_flight = aircraft_flight_line(aircraft_id, new_schedule)
        for index in range(len(aircraft_flight) - 1):
            arrival_airport = aircraft_flight['arrival_airport_code'].iloc[index]
            next_departure_airport = aircraft_flight['departure_airport_code'].iloc[index + 1]
            if arrival_airport != next_departure_airport:
                flag = False
    return flag


def equipment_checker(new_schedule: pd.DataFrame,
                      flight_equipment_table: pd.DataFrame,
                      aircraft_equipment_table: pd.DataFrame) -> bool:
    """Проверяет, подходит ли оснащение самолёта для его рейсов"""
    flag = True
    aircraft_ids = aircraft_equipment_table['aircraft_id'].tolist()
    for aircraft_id in aircraft_ids:
        aircraft_equipment = aircraft_equipment_table[aircraft_equipment_table['aircraft_id'] == aircraft_id]['equipment_id'].iloc[0]
        aircraft_flights = aircraft_flight_line(aircraft_id, new_schedule)
        for index, flight_id in aircraft_flights['flight_id'].items():
            flight_equipment = flight_equipment_table[flight_equipment_table['flight_id'] == flight_id]['equipment_ids'].iloc[0]
            flight_equipment = list(ast.literal_eval(flight_equipment))
            if aircraft_equipment not in flight_equipment:
                flag = False
    return flag


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


def allowed_time_differences(aircraft_flights: pd.DataFrame, time_size: timedelta, allowed_times: list):
    """Проверяет, есть ли необходимый диапазон времени в перерывах между переданными рейсами конкретного ВС,
    в параметр allowed_times добавляются временные промежутки когда может быть проведено ТО"""
    flag = False
    for index in range(len(aircraft_flights.index) - 1):
        arrival_time = pd.to_datetime(aircraft_flights['arrival_time']).iloc[index]
        next_departure_time = pd.to_datetime(aircraft_flights['departure_time']).iloc[index + 1]
        time_delta = next_departure_time - arrival_time
        if time_delta >= time_size:
            allowed_times.append([arrival_time, next_departure_time])
            flag = True
    return flag


def checking_allowed_ts_time(technical_service_table: pd.DataFrame, ts_allowed_time: list, ts_used_time: list) -> int:
    ts_times = list(zip(pd.to_datetime(technical_service_table['time_start'], dayfirst=True),
                        pd.to_datetime(technical_service_table['time_finish'], dayfirst=True)))
    flag = False
    count = 0
    for allowed_time in ts_allowed_time:
        if (allowed_time[0], allowed_time[1]) in ts_times:
            count += 1
            if allowed_time not in ts_used_time:
                flag = True
                ts_allowed_time.remove(allowed_time)
                ts_used_time.append(allowed_time)
                break
    # Штраф если flag = False
    if not flag:
        count = 1000
    return count


def ts_time_shifts(new_schedule: pd.DataFrame, technical_service_table: pd.DataFrame) -> int:
    """Суммарное количество сдвинутых ТО -> min"""
    technical_service_table = remake_technical_service_table(technical_service_table, new_schedule)
    technical_service_aircraft_ids = technical_service_table['aircraft_id'].unique().tolist()
    summary_count = 0
    for aircraft_id in technical_service_aircraft_ids:
        aircraft_flights = aircraft_flight_line(aircraft_id, new_schedule)
        aircraft_ts = technical_service_table[technical_service_table['aircraft_id'] == aircraft_id]
        technical_service_ids = aircraft_ts['technical_service_id'].unique().tolist()
        ts_used_time = []
        ts_allowed_time = []
        for technical_service_id in technical_service_ids:
            time_size = aircraft_ts[aircraft_ts['technical_service_id'] == technical_service_id]['time_size'].iloc[0]
            time_size = pd.to_timedelta(time_size)
            if not allowed_time_differences(aircraft_flights, time_size, ts_allowed_time):
                # Штраф за невозможность проведения ТО
                summary_count += 1000
            else:
                count = checking_allowed_ts_time(technical_service_table, ts_allowed_time, ts_used_time)
                summary_count += count
    return summary_count


def move_parts(trigger_aircraft: int, health_aircraft: int, trigger_part: list, health_part: list,
               partition_list: list) -> list:
    """Меняет местами назначения для переданных ВС в переданных связках, возвращает новое разбиение на связки"""
    for part in partition_list:
        if part == trigger_part:
            for flight in part:
                flight['aircraft_id'] = health_aircraft
        if part == health_part:
            for flight in part:
                flight['aircraft_id'] = trigger_aircraft
    return partition_list


def smart_swap(trigger_aircraft: int,
               health_aircraft: int,
               partition_list: list,
               nearest_flights: pd.DataFrame,
               trigger_aircraft_ids: list,
               trigger_flight_id: str,
               flight_equipment_table: pd.DataFrame,
               aircraft_equipment_table: pd.DataFrame) -> pd.DataFrame:
    """Делаем swap между ВС по связкам пока не будут удовлетворены все штрафные функции"""
    if trigger_aircraft not in trigger_aircraft_ids and health_aircraft in trigger_aircraft_ids:
        trigger_aircraft, health_aircraft, = health_aircraft, trigger_aircraft
    elif health_aircraft not in trigger_aircraft_ids and trigger_aircraft not in trigger_aircraft_ids:
        return nearest_flights

    trigger_timerange = extract_trigger_time_range(trigger_aircraft, trigger_flight_id, partition_list)
    trigger_part = extract_part_using_flight_id(trigger_aircraft, trigger_flight_id, partition_list)
    health_part = extract_part_from_timerange(health_aircraft, nearest_flights, trigger_timerange, partition_list)
    temp_partition = move_parts(trigger_aircraft, health_aircraft, trigger_part, health_part, partition_list)

    if health_part is None or trigger_part is None:
        print(f'Одна из связок пустая')
        # if penalty_function(from_partition_to_dataframe(temp_partition), flight_equipment_table, aircraft_equipment_table) != 0:
        #     print(f'Расписание не удовлетворяет штрафной функции')
        return from_partition_to_dataframe(temp_partition)

    empty_prev_flag = False
    good_prev_partition = False
    empty_next_flag = False
    good_next_partition = False

    temp_trigger_part_for_find_prev = trigger_part
    temp_health_part_for_find_prev = health_part
    temp_trigger_part_for_find_next = trigger_part
    temp_health_part_for_find_next = health_part
    pd_for_test = from_partition_to_dataframe(temp_partition)

    # Для проверки новых связок триггерного рейса влево
    ind_trigger_last = pd_for_test.loc[(pd_for_test['aircraft_id'] == trigger_aircraft) & (
                pd_for_test['previous_solution_id'] == health_part[-1]['previous_solution_id'])].index[0]
    # Для проверки новых связок триггерного рейса вправо
    ind_trigger_first = pd_for_test.loc[(pd_for_test['aircraft_id'] == trigger_aircraft) & (
                pd_for_test['previous_solution_id'] == health_part[0]['previous_solution_id'])].index[0]
    # Для проверки новых связок здорового рейса влево
    ind_health_last = pd_for_test.loc[(pd_for_test['aircraft_id'] == health_aircraft) & (
                pd_for_test['previous_solution_id'] == trigger_part[-1]['previous_solution_id'])].index[0]
    # Для проверки новых связок здорового рейса вправо
    ind_health_first = pd_for_test.loc[(pd_for_test['aircraft_id'] == health_aircraft) & (
                pd_for_test['previous_solution_id'] == trigger_part[0]['previous_solution_id'])].index[0]

    while True:
        trigger_prev_part = extract_next_or_previous_part(trigger_aircraft, temp_trigger_part_for_find_prev,
                                                          partition_list, 'prev')
        health_prev_part = extract_next_or_previous_part(health_aircraft, temp_health_part_for_find_prev,
                                                         partition_list, 'prev')
        if trigger_prev_part != [] and health_prev_part != [] and good_prev_partition is False:
            temp_partition = move_parts(trigger_aircraft, health_aircraft, trigger_prev_part, health_prev_part,
                                        temp_partition)
            temp_trigger_part_for_find_prev = trigger_prev_part
            temp_health_part_for_find_prev = health_prev_part

            pd_for_test = from_partition_to_dataframe(temp_partition)
            pd_trigger_for_test = pd_for_test.loc[(pd_for_test['aircraft_id'] == trigger_aircraft) & (pd_for_test.index <= ind_trigger_last)]
            pd_health_for_test = pd_for_test.loc[ (pd_for_test['aircraft_id'] == health_aircraft) & (pd_for_test.index <= ind_health_last)]
            prev_penalty_flag = (penalty_function(pd_trigger_for_test, flight_equipment_table, aircraft_equipment_table) == 0
                                 and penalty_function(pd_health_for_test, flight_equipment_table, aircraft_equipment_table) == 0)
            if prev_penalty_flag:
                good_prev_partition = True
        else:
            empty_prev_flag = True

        trigger_next_part = extract_next_or_previous_part(trigger_aircraft, temp_trigger_part_for_find_next,
                                                          partition_list, 'next')
        health_next_part = extract_next_or_previous_part(health_aircraft, temp_health_part_for_find_next,
                                                         partition_list, 'next')
        if trigger_next_part != [] and health_next_part != [] and good_next_partition is False:
            temp_partition = move_parts(trigger_aircraft, health_aircraft, trigger_next_part, health_next_part,
                                        temp_partition)
            temp_trigger_part_for_find_next = trigger_next_part
            temp_health_part_for_find_next = health_next_part

            pd_for_test = from_partition_to_dataframe(temp_partition)
            pd_trigger_for_test = pd_for_test.loc[(pd_for_test['aircraft_id'] == trigger_aircraft) & (pd_for_test.index >= ind_trigger_first)]
            pd_health_for_test = pd_for_test.loc[(pd_for_test['aircraft_id'] == health_aircraft) & (pd_for_test.index >= ind_health_first)]
            next_penalty_flag = (penalty_function(pd_trigger_for_test, flight_equipment_table, aircraft_equipment_table) == 0
                                 and penalty_function(pd_health_for_test, flight_equipment_table, aircraft_equipment_table) == 0)
            if next_penalty_flag:
                good_next_partition = True
        else:
            empty_next_flag = True

        if good_next_partition and good_prev_partition:
            break

        if empty_next_flag is True or empty_prev_flag is True:
            print(f'Одна из частей стала пустой')
            break

    return from_partition_to_dataframe(temp_partition)


def swap(trigger_aircraft: int,
         health_aircraft: int,
         partition_list: list,
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

    new_partition = move_parts(trigger_aircraft, health_aircraft, trigger_part, health_part, partition_list)

    new_schedule = from_partition_to_dataframe(new_partition)
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
    return diff


def penalty_function(new_schedule: pd.DataFrame,
                     flight_equipment_table: pd.DataFrame,
                     aircraft_equipment_table: pd.DataFrame) -> int:
    penalty = 1000
    if not (schedule_time_checker(new_schedule)
            and airports_checker(new_schedule)
            and equipment_checker(new_schedule, flight_equipment_table, aircraft_equipment_table)):
        return penalty
    return 0


def objective_function(previous_schedule: pd.DataFrame,
                       new_schedule: pd.DataFrame,
                       flight_equipment_table: pd.DataFrame,
                       aircraft_equipment_table: pd.DataFrame,
                       technical_service_table: pd.DataFrame) -> int:
    return (schedule_differences(previous_schedule, new_schedule)
            + ts_time_shifts(new_schedule, technical_service_table)
            + penalty_function(new_schedule, flight_equipment_table, aircraft_equipment_table))


curr_time = datetime(2025, 1, 22, 0, 0)
nearest_sched = nearest_flights_selection(previous_solution, curr_time, 'KJA', 'LED')
# nearest_sched.to_csv('csv_files/nearest_schedule.csv', index=False, sep=';')
parts = base_airports_partition(nearest_sched, 'KJA', 'LED')

equipment_disrupted_list = equipment_disrupted_flights(flight_equipments, nearest_sched, problematic_aircraft_equipment, curr_time, 0, 1, 2, 3, 4)
flight_shift_disrupted_list = flight_shift_disrupted_flights(previous_solution, problematic_flight_shift, 0, 1, 2, 3)
disrupted_flights = disrupted_flights_for_aircraft_id(13, equipment_disrupted_list, flight_shift_disrupted_list)
# print(disrupted_flights)

trigger_aircraft_list = (extract_trigger_aircraft_ids(problematic_aircraft_equipment) +
                         extract_trigger_aircraft_ids(problematic_flight_shift))

# new_nearest_sched = swap(3, 6, parts, nearest_sched, trigger_aircraft_list, 'FV6516')
# new_nearest_sched.to_csv('csv_files/new_schedule.csv', index=False, sep=';')
# new_nearest_sched_2 = smart_swap(5, 4, parts, nearest_sched, trigger_aircraft_list, 'FV2452', flight_equipments, aircraft)
# new_nearest_sched_2.to_csv('csv_files/new_schedule_2.csv', index=False, sep=';')

# new_partition.to_csv('csv_files/new_schedule_RIGHT_TEST.csv', index=False, sep=';')

# new_partition = swap(3, 6, parts, nearest_sched, trigger_aircraft_list, 'FV6878')
# print(ts_time_shifts(new_partition, technical_service) + schedule_differences(nearest_sched,
#                                                                               new_partition) + penalty_function(
#     new_partition, flight_equipments, aircraft))
# new_partition.to_csv('csv_files/new_schedule_WRONG_TEST.csv', index=False, sep=';')
