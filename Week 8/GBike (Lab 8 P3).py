import numpy as np
from scipy.stats import poisson
from functools import lru_cache

RATE_RENT_A = 3
RATE_RENT_B = 4
RATE_RET_A = 3
RATE_RET_B = 2

GAMMA = 0.9
CAP = 20
SHIFT_LIM = 5

INCOME_PER_RENT = 10
SHIFT_FEE = 2
FREE_SHIFT = 0

PARK_THRESHOLD = 10
PARK_PENALTY = 4

values = np.zeros((CAP + 1, CAP + 1))
policy_map = np.zeros((CAP + 1, CAP + 1), dtype=int)

gap = float("inf")

pmf_rA = [poisson.pmf(i, RATE_RENT_A) for i in range(CAP + 1)]
pmf_rB = [poisson.pmf(i, RATE_RENT_B) for i in range(CAP + 1)]
pmf_Ra = [poisson.pmf(i, RATE_RET_A) for i in range(CAP + 1)]
pmf_Rb = [poisson.pmf(i, RATE_RET_B) for i in range(CAP + 1)]


@lru_cache(None)
def calc_EV(xA, xB, move):
    ev = 0

    max_rA = min(xA - move, CAP)
    max_rB = min(xB + move, CAP)

    for rA in range(max_rA + 1):
        for rB in range(max_rB + 1):

            useA = min(xA - move, rA)
            useB = min(xB + move, rB)

            max_retA = useA
            max_retB = useB

            for Ra in range(max_retA + 1):
                for Rb in range(max_retB + 1):

                    nxtA = min(CAP, (xA - move - useA + Ra))
                    nxtB = min(CAP, (xB + move - useB + Rb))

                    total_income = (useA + useB) * INCOME_PER_RENT

                    if move > 0:            # B → A
                        shift_cost = move * SHIFT_FEE
                    elif move < 0:          # A → B
                        shift_cost = abs(move - 1) * SHIFT_FEE + FREE_SHIFT
                    else:
                        shift_cost = 0

                    if nxtA > PARK_THRESHOLD:
                        shift_cost += PARK_PENALTY
                    if nxtB > PARK_THRESHOLD:
                        shift_cost += PARK_PENALTY

                    net_reward = total_income - shift_cost

                    prob = (
                        pmf_rA[rA]
                        * pmf_rB[rB]
                        * pmf_Ra[Ra]
                        * pmf_Rb[Rb]
                    )

                    ev += prob * (net_reward + GAMMA * values[nxtA, nxtB])

    return ev


iteration = 0
while gap > 1e-6:
    iteration += 1
    gap = 0
    print(f"\nIteration {iteration}: updating grid...")

    for sA in range(CAP + 1):
        for sB in range(CAP + 1):

            prev_val = values[sA, sB]

            best_val = -np.inf
            best_move = 0

            print(f"  Evaluating ({sA}, {sB})...")

            for mv in range(-SHIFT_LIM, SHIFT_LIM + 1):
                if 0 <= sA - mv <= CAP and 0 <= sB + mv <= CAP:
                    ev_outcome = calc_EV(sA, sB, mv)

                    if ev_outcome > best_val:
                        best_val = ev_outcome
                        best_move = mv

            values[sA, sB] = best_val
            policy_map[sA, sB] = best_move

            gap = max(gap, abs(prev_val - best_val))

            print(f"    Optimal move → {best_move}, EV = {best_val:.4f}")

    print(f"Iteration {iteration} finished. Δ = {gap:.6f}")


print("\nOptimal policy computed.\n")
print(policy_map)
