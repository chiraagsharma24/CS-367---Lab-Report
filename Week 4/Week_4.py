import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import os
from PIL import Image

def read_octave_matrix(path):
    nums = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            parts = line.split()
            for p in parts:
                nums.append(int(p))

    rows, cols = nums[0], nums[1]
    pixel_vals = np.array(nums[2:], dtype=np.uint8)

    return pixel_vals.reshape(rows, cols)

# open image in grayscale
def open_image_file(fname):
    if fname.endswith('.mat'):
        return read_octave_matrix(fname)
        im = Image.open(fname).convert('L')
        return np.array(im)

# find the scrambled file 
def locate_scrambled_file():
    wanted = 'Week 4/scrambled_lena.mat'
    if os.path.exists(wanted):
        print("Found scrambled:", wanted)
        return open_image_file(wanted)
    raise FileNotFoundError(f"File not found: {wanted}")

# split image into tiles 
def split_into_tiles(img, grid=8):
    h, w = img.shape
    th = h // grid
    tw = w // grid
    tiles = []
    for i in range(grid):
        for j in range(grid):
            tiles.append(img[i*th:(i+1)*th, j*tw:(j+1)*tw])
    return tiles, th, tw

# join tiles back into image
def join_tiles(tiles, order, grid, th, tw):
    out = np.zeros((grid*th, grid*tw), dtype=np.uint8)
    for pos, tile_id in enumerate(order):
        r = pos // grid
        c = pos % grid
        out[r*th:(r+1)*th, c*tw:(c+1)*tw] = tiles[tile_id]
    return out

# difference between two edges
def edge_score(e1, e2):
    diff = np.abs(e1.astype(float) - e2.astype(float))
    l1 = np.sum(diff)
    l2 = math.sqrt(np.sum(diff**2))
    return l1 + 0.5 * l2

# build pairwise costs for right and bottom adjacency
def make_cost_tables(tiles):
    n = len(tiles)
    right_cost = np.zeros((n, n))
    bottom_cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            right_cost[i][j] = edge_score(tiles[i][:, -1], tiles[j][:, 0])
            bottom_cost[i][j] = edge_score(tiles[i][-1, :], tiles[j][0, :])
    return right_cost, bottom_cost

# total mismatch for full arrangement
def total_mismatch(order, grid, right_cost, bottom_cost):
    total = 0.0
    n = len(order)
    for idx in range(n):
        r = idx // grid
        c = idx % grid
        cur = order[idx]
        if c < grid - 1:
            total += right_cost[cur][order[idx + 1]]
        if r < grid - 1:
            total += bottom_cost[cur][order[idx + grid]]
    return total

# simple greedy fill used as a starting point
def quick_fill(n_tiles, grid, right_cost, bottom_cost, start_seed=0):
    arrangement = [start_seed]
    left = set(range(n_tiles))
    left.remove(start_seed)
    for pos in range(1, n_tiles):
        r = pos // grid
        c = pos % grid
        best = None
        best_c = float('inf')
        for t in left:
            cost = 0.0
            cnt = 0
            if c > 0:
                left_neighbor = arrangement[pos - 1]
                cost += right_cost[left_neighbor][t]
                cnt += 1
            if r > 0:
                top_neighbor = arrangement[pos - grid]
                cost += bottom_cost[top_neighbor][t]
                cnt += 1
            if cnt > 0:
                cost = cost / cnt
            if cost < best_c:
                best_c = cost
                best = t
        arrangement.append(best)
        left.remove(best)
    return arrangement

# annealing-based imprvement (swap two tiles randomly)
def annealing(init_order, grid, right_cost, bottom_cost,
                      temp=20000.0, cooling=0.99985, max_iters=20000):
    cur = init_order.copy()
    best = cur.copy()
    cur_e = total_mismatch(cur, grid, right_cost, bottom_cost)
    best_e = cur_e
    for it in range(max_iters):
        i, j = random.sample(range(len(cur)), 2)
        neigh = cur.copy()
        neigh[i], neigh[j] = neigh[j], neigh[i]
        neigh_e = total_mismatch(neigh, grid, right_cost, bottom_cost)
        delta = neigh_e - cur_e
        if delta < 0 or random.random() < math.exp(-delta / temp):
            cur = neigh
            cur_e = neigh_e
            if cur_e < best_e:
                best = cur.copy()
                best_e = cur_e
        temp *= cooling
    return best, best_e

# run full solver using multiple starts and annealing restarts
def run_solver(tiles, grid, right_cost, bottom_cost):
    n = len(tiles)
    best_global = None
    best_global_e = float('inf')

    tries = min(n, 12)
    for seed in range(tries):
        g = quick_fill(n, grid, right_cost, bottom_cost, seed)
        e = total_mismatch(g, grid, right_cost, bottom_cost)
        if e < best_global_e:
            best_global = g
            best_global_e = e

    best_from_greedy, e1 = annealing(best_global, grid, right_cost, bottom_cost, max_iters=25000)

    rand_init = list(range(n))
    random.shuffle(rand_init)
    best_rand, e2 = annealing(rand_init, grid, right_cost, bottom_cost, temp=25000.0, max_iters=15000)

    if e2 < e1:
        return best_rand, e2
    return best_from_greedy, e1

def run():
    print("Jigsaw solver")

    img = locate_scrambled_file()

    print("Image size:", img.shape)
    grid_size = 8
    tiles, tile_h, tile_w = split_into_tiles(img, grid_size)
    print("Tiles created:", len(tiles), f"({grid_size}x{grid_size})")
    print("Tile size:", tile_h, "x", tile_w)

    right_cost, bottom_cost = make_cost_tables(tiles)

    print("Solving... ")
    t0 = time.time()
    sol_order, sol_energy = run_solver(tiles, grid_size, right_cost, bottom_cost)
    t_elapsed = time.time() - t0
    print("Done. time:", f"{t_elapsed:.1f}s")

    solved_img = join_tiles(tiles, sol_order, grid_size, tile_h, tile_w)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Scrambled')
    ax[0].axis('off')

    ax[1].imshow(solved_img, cmap='gray')
    ax[1].set_title('Solved')
    ax[1].axis('off')

    plt.tight_layout()

    out_path = 'Week 4/puzzle_solved.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print("Saved solution to:", out_path)

    plt.show()

if __name__ == '__main__':
    random.seed(0)
    run()