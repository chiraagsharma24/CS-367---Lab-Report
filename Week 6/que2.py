import numpy as np
import matplotlib.pyplot as plt
# Initialize the board with rooks placed randomly.
def create_random_configuration(n=8):
    layout = np.zeros((n, n), dtype=int)
    chosen = np.random.permutation(n*n)[:n]
    for x in chosen:
        i, j = divmod(x, n)
        layout[i, j] = 1
    return layout

# Calculate the energy of the board configuration.
def compute_cost(state):
    rows = np.sum(state, axis=1)
    cols = np.sum(state, axis=0)
    return np.sum((rows - 1)**2) + np.sum((cols - 1)**2)

# Optimize the board configuration to minimize energy
def minimize_cost(state, steps=1200):
    current = state.copy()
    n = current.shape[0]
    best_cost = compute_cost(current)
    for _ in range(steps):
        a, b = np.random.choice(n*n, 2, replace=False)
        ra, ca = divmod(a, n)
        rb, cb = divmod(b, n)
        if current[ra, ca] == 1 and current[rb, cb] == 0:
            current[ra, ca], current[rb, cb] = current[rb, cb], current[ra, ca]
            new_cost = compute_cost(current)
            if new_cost < best_cost:
                best_cost = new_cost
            else:
                current[ra, ca], current[rb, cb] = current[rb, cb], current[ra, ca]

    return current, best_cost

# Initialize and optimize the board
start_state = create_random_configuration()
print("Initial cost:", compute_cost(start_state))
plt.imshow(start_state, cmap="binary")
plt.title("Initial Board")
plt.show()

final_state, cost = minimize_cost(start_state)
print("Final minimized cost:", cost)
plt.imshow(final_state, cmap="binary")
plt.title("Final Board(Minimized)")
plt.show()
