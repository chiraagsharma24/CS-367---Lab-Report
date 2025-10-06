import random
from Week3_1 import generate_k_sat_problem

# Helper functions 
def heuristic1(sol, clauses):
    count = 0
    for c in clauses:
        sat = False
        for lit in c:
            if lit > 0 and sol[abs(lit)-1]:
                sat = True
            if lit < 0 and not sol[abs(lit)-1]:
                sat = True
        if sat:
            count += 1
    return count

def heuristic2(sol, clauses):
    return heuristic1(sol, clauses) / len(clauses)
def random_solution(n):
    return [random.choice([True, False]) for _ in range(n)]

def flip_neighbors(sol, flip=1):
    n = len(sol)
    nbrs = []
    if flip==1:
        for i in range(n):
            nb = sol[:]
            nb[i] = not nb[i]
            nbrs.append(nb)
    elif flip==2:
        for i in range(n):
            for j in range(i+1, n):
                nb = sol[:]
                nb[i] = not nb[i]
                nb[j] = not nb[j]
                nbrs.append(nb)
    elif flip==3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    nb = sol[:]
                    nb[i] = not nb[i]
                    nb[j] = not nb[j]
                    nb[k] = not nb[k]
                    nbrs.append(nb)
    return nbrs

# Hill-Climbing search algorithm
def hill_climbing(clauses, steps=1000, heuristic=heuristic1):
    random.seed(0)
    n = max(abs(lit) for c in clauses for lit in c)
    sol = random_solution(n)
    sc = heuristic(sol, clauses)
    for _ in range(steps):
        nbrs = flip_neighbors(sol)
        scores = [heuristic(nb, clauses) for nb in nbrs]
        best = max(scores)
        if best <= sc:
            break
        sol = nbrs[scores.index(best)]
        sc = best
        if best==len(clauses):
            break
    return sol, sc

# Beam Serch algorithm
def beam_search(clauses, width=3, steps=100, heuristic=heuristic1):
    random.seed(0)
    n = max(abs(lit) for c in clauses for lit in c)
    beam_list = [random_solution(n) for _ in range(width)]
    for _ in range(steps):
        all_nbrs = []
        for s in beam_list:
            all_nbrs.extend(flip_neighbors(s))
        scored = [(heuristic(nb, clauses), nb) for nb in all_nbrs]
        scored.sort(key=lambda x: x[0], reverse=True)
        beam_list = [x[1] for x in scored[:width]]
        for s in beam_list:
            if heuristic(s, clauses)==len(clauses):
                return s, len(clauses)

    best = max(beam_list, key=lambda x: heuristic(x, clauses))
    return best, heuristic(best, clauses)

# Variable Neighborhood Descent algorithm
def vnd(clauses, steps=1000, heuristic=heuristic1):
    random.seed(0)
    n = max(abs(lit) for c in clauses for lit in c)
    sol = random_solution(n)
    sc = heuristic(sol, clauses)
    step =0
    while step < steps:
        improved = False
        for f in [1,2,3]:
            nbrs = flip_neighbors(sol, flip=f)
            best_sc = sc
            best_sol = sol
            for nb in nbrs:
                s = heuristic(nb, clauses)
                if s > best_sc:
                    best_sc = s
                    best_sol = nb
                    improved = True
            sol, sc = best_sol, best_sc
            if sc==len(clauses):
                return sol, sc
            if improved:
                step += 1
                break
        if not improved:
            break
    return sol, sc


if __name__=="__main__":
    problem_sizes = [(3,7,5)]
    for k, m, n in problem_sizes:
        print(f"\nProblem: k={k}, m={m}, n={n}")
        clauses = generate_k_sat_problem(k, m, n)
        print("clauses:", clauses)

        # VND heuristic1
        sol, sc = vnd(clauses, heuristic=heuristic1)
        print("\nVND Heuristic1: satisfied clauses:", sc, "/",len(clauses))
        print("solution:", sol)

        # Beam width 3 heuristic1
        sol, sc = beam_search(clauses, width=3, heuristic=heuristic1)
        print("\nbeam width 3 Heuristic1: satisfied clauses:", sc, "/",len(clauses))
        print("solution:", sol)

        # Beam width 4 heuristic1
        sol, sc = beam_search(clauses, width=4, heuristic=heuristic1)
        print("\nbeam width 4 Heuristic1: satisfied clauses:", sc, "/",len(clauses))
        print("solution:", sol)

        # Hill Climbing heuristic1
        sol, sc = hill_climbing(clauses, heuristic=heuristic1)
        print("\nhill climbing Heuristic1: satisfied clauses:", sc, "/",len(clauses))
        print("solution:", sol)

        # VND heuristic2
        sol, sc = vnd(clauses, heuristic=heuristic2)
        print("\nVND Heuristic2: satisfied clauses (normalized):", round(sc,2), "/",len(clauses))
        print("solution:", sol)

        # Beam width 3 heuristic2
        sol, sc = beam_search(clauses, width=3, heuristic=heuristic2)
        print("\nbeam width 3 Heuristic2: satisfied clauses (normalized):", round(sc,2), "/",len(clauses))
        print("solution:", sol)

        # Beam width 4 heuristic2
        sol, sc = beam_search(clauses, width=4, heuristic=heuristic2)
        print("\nbeam width 4 Heuristic2: satisfied clauses (normalized):", round(sc,2), "/",len(clauses))
        print("solution:", sol)
