import random
import numpy as np
import matplotlib.pyplot as plt

cities = {
    'A': (0, 0),
    'B': (1, 5),
    'C': (5, 2),
    'D': (3, 6),
    'E': (8, 3),
    'F': (2, 4),
    'G': (6, 7)
}

POPULATION_SIZE = 50
MUTATION_RATE = 0.1
ELITE_SIZE = 5
GENERATIONS = 100

def calculate_distance(route):
    total = 0
    for i in range(len(route)):
        city_a = cities[route[i]]
        city_b = cities[route[(i+1) % len(route)]]
        total += np.sqrt((city_a[0]-city_b[0])**2 + (city_a[1]-city_b[1])**2)
    return total

def fitness(route):
    return 1 / calculate_distance(route)

def create_individual():
    route = list(cities.keys())
    random.shuffle(route)
    return route

def initialize_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def tournament_selection(population, k=3):
    tournament = random.sample(population, k)
    return max(tournament, key=lambda x: fitness(x))

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    child = [None]*size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    remaining = [city for city in parent2 if city not in child]
    ptr = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[ptr]
            ptr += 1
    return child

def swap_mutation(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm():
    population = initialize_population()
    best_distance = float('inf')
    best_route = None
    progress = []
    for generation in range(GENERATIONS):
        population = sorted(population, key=lambda x: fitness(x), reverse=True)
        current_best = population[0]
        current_dist = calculate_distance(current_best)
        if current_dist < best_distance:
            best_distance = current_dist
            best_route = current_best.copy()
        progress.append(best_distance)
        if generation % 10 == 0:
            print(f"Gen {generation}: Best Distance = {best_distance:.2f}")
        next_population = population[:ELITE_SIZE]
        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = ordered_crossover(parent1, parent2)
            child = swap_mutation(child)
            next_population.append(child)
        population = next_population
    return best_route, best_distance, progress

best_route, best_distance, progress = genetic_algorithm()

print("\n=== Optimal Route ===")
print(" → ".join(best_route))
print(f"Total Distance: {best_distance:.2f}")

plt.plot(progress)
plt.title('Optimization Progress')
plt.xlabel('Generation')
plt.ylabel('Best Distance')
plt.show()

x = [cities[city][0] for city in best_route + [best_route[0]]]
y = [cities[city][1] for city in best_route + [best_route[0]]]
plt.plot(x, y, 'o-')
plt.title('Optimal Route')
for city, (xi, yi) in cities.items():
    plt.text(xi, yi, city)
plt.show()
