import math
import random
import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(point1, point2):
    sum_squared_diff = sum((x - y) ** 2 for x, y in zip(point1, point2))
    return math.sqrt(sum_squared_diff)


def calculate_distance_matrix(file, amount_of_citys=52):
    with open(file) as file:
        coord_list = []
        # city_counter = 0
        for line in file:
            x_1, x_2 = list(map(int, line.replace('.0', '').split()))
            coord_list.append([x_1, x_2])
            # city_counter += 1
    distance_matrix = np.zeros((amount_of_citys, amount_of_citys))
    for city_1 in range(len(coord_list)):
        for city_2 in range(len(coord_list)):
            distance_matrix[city_1, city_2] = euclidean_distance(coord_list[city_1], coord_list[city_2])

    return distance_matrix


def inverse(permutation, i, j):
    """inverse the cities between positions i and j"""
    result_permutation = permutation.copy()
    if not (0 <= i and j <= len(permutation) and 0 <= j - i < len(permutation) - 1):
        return result_permutation

    result_permutation[i:j+1] = reversed(result_permutation[i:j+1])
    return result_permutation


def insert(permutation, i, j):
    """move the city in position j to position i"""
    result_permutation = permutation.copy()
    if i < j:
        result_permutation[i+1:j+1], result_permutation[i] = result_permutation[i:j], result_permutation[j]
    if i > j:
        result_permutation[j:i], result_permutation[i] = result_permutation[j+1:i+1], result_permutation[j]
    return result_permutation


def swap(permutation, i, j):
    """swap the city in position j and city in position i"""
    result_permutation = permutation.copy()
    result_permutation[i], result_permutation[j] = result_permutation[j], result_permutation[i]
    return result_permutation


distances_matrix = calculate_distance_matrix('csv_files/berlin52.txt')


def calculate_distance_of_permutation(permutation: list):
    total_distance = 0
    num_cities = len(permutation)

    for i in range(num_cities - 1):
        total_distance += distances_matrix[permutation[i]][permutation[i + 1]]

    # Замыкаем маршрут (возвращаемся в начальный город)
    total_distance += distances_matrix[permutation[-1]][permutation[0]]

    return total_distance


def hybrid_operator(permutation, i, j, op_1, op_2, op_3):
    return min(op_1(permutation, i, j), op_2(permutation, i, j), op_3(permutation, i, j),
               key=lambda x: calculate_distance_of_permutation(x))


def generate_candidate_solution(current_solution):
    i, j = random.sample(range(len(current_solution)), 2)
    return hybrid_operator(current_solution, i, j, inverse, insert, swap)


def list_based_sa_algorithm(temperature_list, max_iteration_times, markov_chain_length, amount_of_citys=52):
    outer_loop_iterator = 0
    # Generate an initial solution x randomly
    current_solution = [i for i in range(amount_of_citys)]
    random.shuffle(current_solution)
    # result_list = []
    # probability_list = []
    # exponent_list = []
    # temperature_list_for_plot = []
    # objective_functions_list = []

    while outer_loop_iterator <= max_iteration_times:
        temperature_max = max(temperature_list)
        # temperature_list_for_plot.append(temperature_max)
        outer_loop_iterator += 1
        temperature = 0
        bad_solution_count, inner_loop_iterator = 0, 0
        while inner_loop_iterator <= markov_chain_length:
            # Generate a candidate solution y randomly
            # based on current solution x and a specified
            # neighbourhood structure
            candidate_solution = generate_candidate_solution(current_solution)

            inner_loop_iterator += 1
            current_solution_distance_sum = calculate_distance_of_permutation(current_solution)
            candidate_solution_distance_sum = calculate_distance_of_permutation(candidate_solution)

            if candidate_solution_distance_sum < current_solution_distance_sum:
                current_solution = candidate_solution
                probability = 1
            else:
                safe_temperature_max = max(temperature_max, 1e-10)
                exponent = -(candidate_solution_distance_sum - current_solution_distance_sum) / safe_temperature_max
                # exponent_list.append(exponent)
                probability = math.exp(exponent)
                random_float = random.random()
                if random_float < probability:
                    # когда temperature == 0 и candidate_solution_distance_sum == current_solution_distance_sum получается число меньшее 1e-10
                    temperature = max((temperature - candidate_solution_distance_sum + current_solution_distance_sum) / math.log(random_float), 1e-10)
                    # temperature_list_for_plot.append(temperature_max)

                    # if temperature == 1e-10:
                    #     print(prev_temp, candidate_solution_distance_sum - current_solution_distance_sum, math.log(random_float))

                    bad_solution_count += 1
                    current_solution = candidate_solution
                else:
                    pass
            # objective_functions_list.append(current_solution_distance_sum)
            # probability_list.append(probability)
            # result_list.append(calculate_distance_of_permutation(current_solution))

        if bad_solution_count != 0:
            temperature_list.remove(temperature_max)
            # temperature_list_for_plot.append(temperature_max)
            temperature_list.append(temperature / bad_solution_count)

    # print(probability_list)
    # plt.plot(range(5000), temperature_list_for_plot[:5000])
    # plt.plot(range(5000), objective_functions_list[:5000])
    # plt.plot(range(max_iteration_times), probability_list[:max_iteration_times])
    # plt.plot(range(max_iteration_times*markov_chain_length), result_list[:max_iteration_times*markov_chain_length])
    # plt.plot(range(9000), exponent_list[:9000])
    # plt.plot()
    # plt.show()
    return calculate_distance_of_permutation(current_solution)


temperature_list = [1000, 500, 250, 125, 60, 30, 15, 7, 3, 1]  # Пример списка температур

print(list_based_sa_algorithm(temperature_list, 100, 100))