import numpy as np
def _hungarian(cost):
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    n_r, n_c = cost.shape
    n = max(n_r, n_c)
    big = cost.max() + 1.0 if cost.size else 1.0
    C = np.full((n, n), big, dtype=float)
    C[:n_r, :n_c] = cost
    u = np.zeros(n + 1); v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int); way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = np.full(n + 1, np.inf); used = np.zeros(n + 1, dtype=bool)
        count = 0
        while True:
            count += 1
            if count > 100:
                print('HANG!')
                return
            used[j0] = True
            i0 = p[j0]; delta = np.inf; j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, n + 1):
        i = p[j]
        if i <= n_r and j <= n_c:
            rows.append(i - 1); cols.append(j - 1)
    order = np.argsort(rows)
    return np.array(rows)[order], np.array(cols)[order]

_hungarian(np.array([[np.nan, 1.0], [1.0, 1.0]]))
