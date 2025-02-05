import math
import random
import numpy as np


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


def calculate_distance_of_permutation(permutation: list):
    distances_matrix = np.random.randint(1, 15, (5, 5), dtype=int)
    print(distances_matrix)
    result = sum([distances_matrix[i][i + 1] for i in permutation[0:-1]]) + distances_matrix[len(permutation) - 1][permutation[0]]
    return result


def hybrid_operator(permutation, i, j, op_1, op_2, op_3):
    return min(op_1(permutation, i, j), op_2(permutation, i, j), op_3(permutation, i, j),
               key=lambda x: calculate_distance_of_permutation(permutation))


def generate_candidate_solution(current_solution):
    i, j = random.sample(range(len(current_solution)), 2)
    return hybrid_operator(current_solution, i, j, inverse, insert, swap)


def list_based_sa_algorithm(temperature_list, max_iteration_times, markov_chain_length, amount_of_citys):
    outer_loop_iterator = 0
    # Generate an initial solution x randomly
    current_solution = [i for i in range(amount_of_citys)]
    random.shuffle(current_solution)

    while outer_loop_iterator <= max_iteration_times:
        temperature_max = max(temperature_list)
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
            else:
                probability = math.exp(-(candidate_solution_distance_sum - current_solution_distance_sum) / temperature_max)
                random_float = random.random()
                if random_float < probability:
                    temperature = (temperature - candidate_solution_distance_sum + current_solution_distance_sum) / math.log(random_float)
                    bad_solution_count += 1
                    current_solution = candidate_solution
                else:
                    pass

        if bad_solution_count != 0:
            temperature_list.remove(temperature_max)
            temperature_list.append(temperature / bad_solution_count)

    return current_solution

