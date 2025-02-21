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
                disrupted_flights.append((flight_id, aircraft_id, int(previous_solution_id)))

    return disrupted_flights


def equipment_disrupted_flights_checker(flight_equipments_table: pd.DataFrame,
                                        previous_solution_table: pd.DataFrame,
                                        disruptions_table: pd.DataFrame,
                                        current_time: datetime,
                                        *disruption_ids: int) -> bool:
    """Возвращает True если расписание остается верным(не ломается, иначе False"""
    if len(equipment_disrupted_flights(flight_equipments_table,
                                       previous_solution_table,
                                       disruptions_table,
                                       current_time,
                                       *disruption_ids)) > 0:
        return False
    return True


def get_time_from_table(table: pd.DataFrame, previous_solution_id: int, column_name: str) -> datetime:
    if 'new' not in column_name:
        previous_solution_id += 1
    string_time = table[table['previous_solution_id'] == previous_solution_id][column_name].iloc[0]

    time_object = parser.parse(string_time, dayfirst=True)
    return time_object


def flight_shift_disrupted_flights(previous_solution_table: pd.DataFrame,
                                   disruptions_table: pd.DataFrame,
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


def flight_shift_disrupted_flights_checker(previous_solution_table: pd.DataFrame,
                                           disruptions_table: pd.DataFrame,
                                           *disruption_ids: int) -> bool:
    if len(flight_shift_disrupted_flights(previous_solution_table, disruptions_table, *disruption_ids)) > 0:
        return False
    return True


def generate_airport_pairs(previous_solution_table: pd.DataFrame) -> dict:
    """Возвращает список кортежей, в которых хранятся previous_solution_id для соединенных рейсов"""
    departure_airport_row = previous_solution_table['departure_airport_code']
    arrival_airport_row = previous_solution_table['arrival_airport_code']
    previous_solution_id_row = previous_solution_table['previous_solution_id']

    aircraft_id_row = previous_solution_table['aircraft_id'].astype(str)
    airport_pairs_list = {aircraft_id: [] for aircraft_id in aircraft_id_row.unique()}

    for i in range(len(previous_solution_table)):
        flight_dict = {'departure_airport': departure_airport_row.iloc[i],
                       'arrival_airport': arrival_airport_row.iloc[i],
                       'aircraft_id': str(aircraft_id_row.iloc[i]),
                       'previous_solution_id': int(previous_solution_id_row.iloc[i])}
        airport_pairs_list[str(aircraft_id_row.iloc[i])].append(flight_dict)

    return airport_pairs_list


def base_airports_partition(previous_solution_table: pd.DataFrame, *base_airports: str) -> list:
    """LED, KJA. Отдельно обработать полеты из базы в базу"""
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


def flights_objective_function():
    """Возвращаем 1 если переданное расписание корректно(не ломается от """

