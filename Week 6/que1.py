import numpy as np
import matplotlib.pyplot as plt
import random

# Define patterns for letters D, J, C, M
letter_grids = {
    'D': np.array([-1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1]).reshape(5, 5),
    'J': np.array([-1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1]).reshape(5, 5),
    'C': np.array([1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1]).reshape(5, 5),
    'M': np.array([-1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1]).reshape(5, 5)
}

# Show character patterns 
fig1 = plt.figure(figsize=(6,6))
for k, (symbol, mat) in enumerate(letter_grids.items()):
    plt.subplot(1,4,k+1)
    plt.title(symbol)
    plt.axis("off")
    plt.imshow(mat, cmap='gray')
plt.show()


# Weight Computation (Hebbian) 
count_letters = len(letter_grids)
dim = np.prod(next(iter(letter_grids.values())).shape)

W = np.zeros((dim, dim))
for grid in letter_grids.values():
    vec = grid.flatten()
    W += np.outer(vec, vec)

np.fill_diagonal(W, 0)
W /= count_letters


# introduce error 
def distort_pattern(src_grid, flips=1):
    changed = src_grid.copy()
    altered = set()

    while len(altered) < flips:
        r, c = np.random.randint(0, 5, size=2)
        if (r,c) not in altered:
            altered.add((r,c))
            changed[r, c] *= -1
    return changed


# Test and Plot Results
fig2 = plt.figure(figsize=(9,45))

for nerr in range(1, 10+1):

    chosen = random.choice(list(letter_grids.values()))
    damaged = distort_pattern(chosen, nerr)

    state = damaged.flatten()

    last_score = float("inf")
    now = np.linalg.norm(state)

    # recovery loop
    while now < last_score:
        last_score = now
        state = np.sign(W @ state)
        now = np.linalg.norm(state - damaged.flatten())

    # drawing
    plt.subplot(10,3,3*(nerr-1)+1)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(chosen, cmap="gray")

    plt.subplot(10,3,3*(nerr-1)+2)
    plt.title(f"{nerr} noisy")
    plt.axis("off")
    plt.imshow(damaged, cmap="gray")

    plt.subplot(10,3,3*(nerr-1)+3)
    plt.title("Recovered")
    plt.axis("off")
    plt.imshow(state.reshape(5,5), cmap="gray")

plt.show()
