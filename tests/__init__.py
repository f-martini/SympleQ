import random


PRIME_LIST = [2, 3, 5, 7, 11]


def choose_random_dimensions(max_product: int, available_primes: list[int] = PRIME_LIST) -> list[int]:
    if max_product is None:
        max_product = 10 * max(available_primes) + 1

    dimensions = [random.choice(available_primes)]
    prod = dimensions[0]
    while True:
        d = random.choice(available_primes)
        if prod * d >= max_product:
            return dimensions

        dimensions.append(d)
        prod *= d
