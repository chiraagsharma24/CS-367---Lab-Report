from collections import deque
# successor function
def get_next_states(state):        
    next_states = []
    s = list(state)
    idx = s.index('_')             # idx - stores the position of the blank ('_')

    # East rabbit moves right
    if idx > 0 and s[idx-1] == 'E':
        temp = s[:]
        temp[idx], temp[idx-1] = temp[idx-1], temp[idx]
        next_states.append(tuple(temp))

    # East rabbit jumps right
    if idx > 1 and s[idx-2] == 'E':
        temp = s[:]
        temp[idx], temp[idx-2] = temp[idx-2], temp[idx]
        next_states.append(tuple(temp))

    # West rabbit moves left
    if idx < 6 and s[idx+1] == 'W':
        temp = s[:]
        temp[idx], temp[idx+1] = temp[idx+1], temp[idx]
        next_states.append(tuple(temp))

    # West rabbit jumps left
    if idx < 5 and s[idx+2] == 'W':
        temp = s[:]
        temp[idx], temp[idx+2] = temp[idx+2], temp[idx]
        next_states.append(tuple(temp))

    return next_states

def solve_rabbits(method='bfs'):
    start = ('E', 'E', 'E', '_', 'W', 'W', 'W')
    goal = ('W', 'W', 'W', '_', 'E', 'E', 'E')
    q = deque()
    q.append((start, [start]))
    visited = set()
    visited.add(start)
    while q:
        if method == 'bfs':
            cur, pt = q.popleft()           # pt means path , cur means current states
        else:
            cur, pt = q.pop()
        if cur==goal:
            return pt
        nxt = get_next_states(cur)          # nxt means next states 
        if method=='dfs':
            nxt.reverse()
        for n in nxt:
            if n not in visited:
                visited.add(n)
                q.append((n, pt + [n]))
    return None

def print_pt(pt):
    if not pt:
        print("No solution found")
        return
    print("steps count:", len(pt) - 1)
    for st in pt:
        print(" ".join(st))

if __name__ == "__main__":
    print("Rabbit leap Problem")
    print("Initial state : E E E _ W W ")
    print("Final state   : W W W _ E E E ")

    print("\nusing BFS traversal :")
    bfs = solve_rabbits('bfs')
    print_pt(bfs)
    print("\n")

    print("using DFS traversal:")
    dfs = solve_rabbits('dfs')
    print_pt(dfs)
