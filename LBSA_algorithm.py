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
