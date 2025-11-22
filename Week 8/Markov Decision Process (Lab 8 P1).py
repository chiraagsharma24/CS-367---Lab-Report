import numpy as np

GRID_H = 3
GRID_W = 4

GOAL_POS = (0, 3)
TRAP_POS = (1, 3)
INIT_POS = (2, 0)

IS_DET = True


class Env:
    def __init__(self, pos=INIT_POS, win=GOAL_POS, lose=TRAP_POS):
        self.grid = np.zeros((GRID_H, GRID_W))
        self.curr = pos
        self.terminal = False
        self.deterministic = IS_DET
        self.win_pos = win
        self.lose_pos = lose

    def reward_of(self, nxt):
        if nxt == self.win_pos:
            return 1
        elif nxt == self.lose_pos:
            return -1
        return -0.04

    def check_terminal(self):
        if self.curr in [self.win_pos, self.lose_pos]:
            self.terminal = True

    def move(self, pos, act):
        if self.deterministic:
            if act == "up":
                nxt = (pos[0] - 1, pos[1])
            elif act == "down":
                nxt = (pos[0] + 1, pos[1])
            elif act == "left":
                nxt = (pos[0], pos[1] - 1)
            else:
                nxt = (pos[0], pos[1] + 1)

            if 0 <= nxt[0] < GRID_H and 0 <= nxt[1] < GRID_W:
                if nxt != (1, 1):
                    return nxt
        return pos

    def display(self):
        tmp = np.copy(self.grid)
        tmp[self.curr] = 1
        tmp[self.win_pos] = -1
        tmp[self.lose_pos] = -1

        for r in range(GRID_H):
            print("-----------------")
            row = "| "
            for c in range(GRID_W):
                if tmp[r, c] == 1:
                    t = '*'
                elif tmp[r, c] == -1:
                    t = 'X'
                else:
                    t = '0'
                row += t + " | "
            print(row)
        print("-----------------")


class Learner:
    def __init__(self):
        self.history = []
        self.env = Env()

        self.gamma = 0.9
        self.actions = ["up", "down", "left", "right"]

        # value table
        self.v_table = {(r, c): 0 for r in range(GRID_H) for c in range(GRID_W)}
        self.v_table[GOAL_POS] = 1
        self.v_table[TRAP_POS] = -1

    def restart(self):
        self.history = []
        self.env = Env()

    def related_dirs(self, act):
        if act in ["up", "down"]:
            return ["left", "right"]
        return ["up", "down"]

    def estimate(self, pos, act):
        total = 0
        primary_next = self.env.move(pos, act)

        # unintended direction contributions
        for alt in self.related_dirs(act):
            nxt = self.env.move(pos, alt)
            total += 0.1 * self.env.reward_of(nxt)
            total += 0.1 * self.gamma * self.v_table[nxt]

        # intended direction
        total += 0.8 * self.env.reward_of(primary_next)
        total += 0.8 * self.gamma * self.v_table[primary_next]

        return total

    def evaluate(self):
        threshold = 1
        while True:
            diff = 0
            for r in range(GRID_H):
                for c in range(GRID_W):
                    if (r, c) in [GOAL_POS, TRAP_POS]:
                        continue

                    best_val = -np.inf
                    for act in self.actions:
                        est = 0.25 * self.estimate((r, c), act)
                        best_val = max(best_val, est)

                    diff = max(diff, abs(self.v_table[(r, c)] - best_val))
                    self.v_table[(r, c)] = best_val

            if diff < threshold:
                break

    def print_values(self):
        for r in range(GRID_H):
            print("----------------------------------")
            row = "| "
            for c in range(GRID_W):
                row += str(round(self.v_table[(r, c)], 4)).ljust(6) + " | "
            print(row)
        print("----------------------------------")


if __name__ == "__main__":
    env = Env()
    env.display()

    agent = Learner()
    agent.evaluate()
    agent.print_values()
