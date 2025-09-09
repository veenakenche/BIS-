import random


def rk4_step(f, t, y, h, a):
    k1 = f(t, y, a)
    k2 = f(t + h/2, y + h*k1/2, a)
    k3 = f(t + h/2, y + h*k2/2, a)
    k4 = f(t + h, y + h*k3, a)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def ode_model(t, y, a):
    return -a * y

def solve_ode_rk4(f, t0, tf, y0, a, num_steps):
    h = (tf - t0) / num_steps
    t_values = [t0 + i*h for i in range(num_steps + 1)]
    y_values = [y0]
    y = y0
    for i in range(num_steps):
        y = rk4_step(f, t_values[i], y, h, a)
        y_values.append(y)
    return t_values, y_values

def mean_squared_error(y_pred, y_true):
    n = len(y_true)
    total = 0.0
    for i in range(n):
        diff = y_pred[i] - y_true[i]
        total += diff * diff
    return total / n

def objective_function(a, t_eval, target_data):
    t0 = t_eval[0]
    tf = t_eval[-1]
    num_steps = len(t_eval) - 1
    _, y_pred = solve_ode_rk4(ode_model, t0, tf, 1.0, a, num_steps)
    return mean_squared_error(y_pred, target_data)

num_particles = 30
num_iterations = 5
dim = 1
w = 0.5
c1 = 1.5
c2 = 1.5

param_min = 0.1
param_max = 2.0


true_a = 0.7
num_points = 50
t_eval = [i * (5.0 / num_points) for i in range(num_points + 1)]

_, true_solution = solve_ode_rk4(ode_model, 0.0, 5.0, 1.0, true_a, num_points)

target_data = [y + random.gauss(0, 0.05) for y in true_solution]

positions = [[random.uniform(param_min, param_max)] for _ in range(num_particles)]
velocities = [[0.0] for _ in range(num_particles)]
personal_best_positions = [pos[:] for pos in positions]
personal_best_scores = [objective_function(p[0], t_eval, target_data) for p in positions]
global_best_idx = min(range(num_particles), key=lambda i: personal_best_scores[i])
global_best_position = personal_best_positions[global_best_idx][:]
global_best_score = personal_best_scores[global_best_idx]
for iteration in range(num_iterations):
    for i in range(num_particles):
        r1 = random.random()
        r2 = random.random()

        velocities[i][0] = (
            w * velocities[i][0]
            + c1 * r1 * (personal_best_positions[i][0] - positions[i][0])
            + c2 * r2 * (global_best_position[0] - positions[i][0])
        )

        positions[i][0] += velocities[i][0]

        # Clamp
        if positions[i][0] < param_min:
            positions[i][0] = param_min
        elif positions[i][0] > param_max:
            positions[i][0] = param_max

        score = objective_function(positions[i][0], t_eval, target_data)

        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i][:]

            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[i][:]

    print(f"Iteration {iteration + 1}/{num_iterations} - Best MSE: {global_best_score:.6f} - Best param: {global_best_position[0]:.4f}")

print("\nOptimization finished")
print(f"Estimated parameter a: {global_best_position[0]:.4f}")
print(f"True parameter a: {true_a}")
