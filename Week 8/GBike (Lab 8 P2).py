import numpy as np
from scipy.stats import poisson
from functools import lru_cache

LAM_RENT_A = 3
LAM_RENT_B = 4
LAM_RET_A = 3
LAM_RET_B = 2

DISCOUNT = 0.9
MAX_CAP = 20
MAX_SHIFT = 5

RENT_GAIN = 10
SHIFT_COST = 2

value_grid = np.zeros((MAX_CAP + 1, MAX_CAP + 1))
policy_grid = np.zeros((MAX_CAP + 1, MAX_CAP + 1), dtype=int)

convergence_gap = float("inf")

rentA_probs = [poisson.pmf(i, LAM_RENT_A) for i in range(MAX_CAP + 1)]
rentB_probs = [poisson.pmf(i, LAM_RENT_B) for i in range(MAX_CAP + 1)]
retA_probs = [poisson.pmf(i, LAM_RET_A) for i in range(MAX_CAP + 1)]
retB_probs = [poisson.pmf(i, LAM_RET_B) for i in range(MAX_CAP + 1)]

@lru_cache(None)
def expected_return(nA, nB, shift):
    total_ev = 0

    max_rentA = min(nA - shift, MAX_CAP)
    max_rentB = min(nB + shift, MAX_CAP)

    for rA in range(max_rentA + 1):
        for rB in range(max_rentB + 1):
            takeA = min(nA - shift, rA)
            takeB = min(nB + shift, rB)

            max_retA = takeA
            max_retB = takeB

            for retA in range(max_retA + 1):
                for retB in range(max_retB + 1):
                    nextA = min(MAX_CAP, (nA - shift - takeA + retA))
                    nextB = min(MAX_CAP, (nB + shift - takeB + retB))

                    income = (takeA + takeB) * RENT_GAIN
                    movement_penalty = abs(shift) * SHIFT_COST
                    reward = income - movement_penalty

                    prob = (
                        rentA_probs[rA]
                        * rentB_probs[rB]
                        * retA_probs[retA]
                        * retB_probs[retB]
                    )

                    total_ev += prob * (reward + DISCOUNT * value_grid[nextA, nextB])

    return total_ev

iteration = 0
while convergence_gap > 1e-6:
    iteration += 1
    convergence_gap = 0
    print(f"\nIteration {iteration}: updating states...")

    for a in range(MAX_CAP + 1):
        for b in range(MAX_CAP + 1):

            old_val = value_grid[a, b]
            best_act = 0
            best_ev = -np.inf

            print(f"  Evaluating state ({a}, {b})...")

            for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):
                if 0 <= a - shift <= MAX_CAP and 0 <= b + shift <= MAX_CAP:
                    ev = expected_return(a, b, shift)

                    if ev > best_ev:
                        best_ev = ev
                        best_act = shift

            value_grid[a, b] = best_ev
            policy_grid[a, b] = best_act

            convergence_gap = max(convergence_gap, abs(old_val - best_ev))

            print(f"    → Best shift: {best_act}, EV = {best_ev:.4f}")

    print(f"Iteration {iteration} complete. Δ = {convergence_gap:.6f}")

print("\nOptimal policy and value function derived.")

print("\nFinal Movement Policy (A → B):")
print(policy_grid)
