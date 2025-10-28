import random
import math

def sphere_function(position):
    return sum(x ** 2 for x in position)

def get_neighbors(index, grid_size):
    rows, cols = grid_size
    row, col = divmod(index, cols)
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append(nr * cols + nc)
    return neighbors

def update_rule(pos, best_neighbor_pos):
    r = random.random()
    new_pos = [p + r * (bn - p) for p, bn in zip(pos, best_neighbor_pos)]
    return new_pos

def parallel_cellular_algorithm(f, num_cells=9, grid_size=(3, 3), dim=2, max_iterations=5):
    grid = []
    for _ in range(num_cells):
        position = [random.uniform(-5, 5) for _ in range(dim)]
        fitness = f(position)
        grid.append({"position": position, "fitness": fitness})

    best_solution = min(grid, key=lambda c: c["fitness"])

    for iteration in range(max_iterations):
        new_grid = []
        for i in range(num_cells):
            neighbors_idx = get_neighbors(i, grid_size)
            best_neighbor = min([grid[j] for j in neighbors_idx], key=lambda c: c["fitness"])
            new_pos = update_rule(grid[i]["position"], best_neighbor["position"])
            new_fit = f(new_pos)
            if new_fit < grid[i]["fitness"]:
                new_grid.append({"position": new_pos, "fitness": new_fit})
            else:
                new_grid.append(grid[i])

        grid = new_grid
        current_best = min(grid, key=lambda c: c["fitness"])
        if current_best["fitness"] < best_solution["fitness"]:
            best_solution = current_best

        print(f"Iteration {iteration + 1}: Best Fitness = {best_solution['fitness']:.6f}")

    return best_solution

if __name__ == "__main__":
    best = parallel_cellular_algorithm(sphere_function)
    print("\nBest Solution Found:")
    print("Position =", best["position"])
    print("Fitness =", best["fitness"])
