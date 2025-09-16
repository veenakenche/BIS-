import math
import random

cities = {
    0: (0, 0),
    1: (2, 0),
    2: (2, 2),
    3: (0, 2),
    4: (1, 1)
}

num_cities = len(cities)
distance = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
for i in range(num_cities):
    for j in range(num_cities):
        x1, y1 = cities[i]
        x2, y2 = cities[j]
        distance[i][j] = round(math.dist((x1, y1), (x2, y2)), 2)

num_ants = 5
num_iterations = 5
alpha = 1
beta = 5
rho = 0.5
Q = 100
initial_pheromone = 1.0

pheromone = [[initial_pheromone for _ in range(num_cities)] for _ in range(num_cities)]
best_tour_length = float("inf")
best_tour = None

def tour_length(tour):
    length = 0
    for i in range(len(tour) - 1):
        length += distance[tour[i]][tour[i + 1]]
    return length

for iteration in range(num_iterations):
    tours = []
    lengths = []
    for ant in range(num_ants):
        start_city = random.randint(0, num_cities - 1)
        tour = [start_city]
        visited = {start_city}
        current_city = start_city
        while len(visited) < num_cities:
            probabilities = []
            unvisited = [c for c in range(num_cities) if c not in visited]
            for city in unvisited:
                tau = pheromone[current_city][city] ** alpha
                eta = (1 / distance[current_city][city]) ** beta if distance[current_city][city] > 0 else 0
                probabilities.append(tau * eta)
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            r = random.random()
            cumulative = 0
            for idx, city in enumerate(unvisited):
                cumulative += probabilities[idx]
                if r <= cumulative:
                    next_city = city
                    break
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
        tour.append(start_city)
        Lk = tour_length(tour)
        tours.append(tour)
        lengths.append(Lk)
    for i in range(num_cities):
        for j in range(num_cities):
            pheromone[i][j] *= (1 - rho)
    for k, tour in enumerate(tours):
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            pheromone[a][b] += Q / lengths[k]
            pheromone[b][a] += Q / lengths[k]
    min_length = min(lengths)
    if min_length < best_tour_length:
        best_tour_length = min_length
        best_tour = tours[lengths.index(min_length)]
    print(f"Iteration {iteration+1}: Shortest length = {min_length:.2f}")

print("\nBest tour found:", best_tour)
print("Best tour length:", round(best_tour_length, 2))

