import numpy as np
from scipy.optimize import basinhopping, minimize
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inner_polygons", type=int)
parser.add_argument("inner_sides", type=int)
parser.add_argument("container_sides", type=int)
parser.add_argument("--attempts", type=int, default=1000)
parser.add_argument("--tolerance", type=float, default=1e-8)
parser.add_argument("--finalstep", type=float, default=1e-4)
args = parser.parse_args()

N = args.inner_polygons
nsi = args.inner_sides
nsc = args.container_sides
attempts = args.attempts
penalty_tol = args.tolerance
final_step = args.finalstep

def regular_polygon(n):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    verts = np.column_stack((np.cos(angles), np.sin(angles)))
    normals = np.column_stack((np.cos(angles + np.pi/n), np.sin(angles + np.pi/n)))
    return verts, normals

unit_poly, unit_poly_normals = regular_polygon(nsi)
unit_cont, unit_cont_normals = regular_polygon(nsc)
container_apothem = np.cos(np.pi / nsc)

@njit(cache=True)
def transform(x, y, angle, verts):
    c, s = np.cos(angle), np.sin(angle)
    out = np.empty_like(verts)
    for i in range(verts.shape[0]):
        vx, vy = verts[i]
        out[i, 0] = x + vx*c - vy*s
        out[i, 1] = y + vx*s + vy*c
    return out

@njit(cache=True)
def rotate(angle, vecs):
    c, s = np.cos(angle), np.sin(angle)
    out = np.empty_like(vecs)
    for i in range(vecs.shape[0]):
        vx, vy = vecs[i]
        out[i, 0] = vx*c - vy*s
        out[i, 1] = vx*s + vy*c
    return out

@njit(cache=True)
def container_penalty(verts, S):
    limit = container_apothem * S
    penalty = 0.0
    for v in range(verts.shape[0]):
        x, y = verts[v]
        for i in range(nsc):
            d = x * unit_cont_normals[i, 0] + y * unit_cont_normals[i, 1]
            if d > limit:
                diff = d - limit
                penalty += diff * diff
    return penalty

@njit(cache=True)
def sat_overlap(poly1, poly2, axes):
    min_overlap = 1e18
    for ax in axes:
        axx, axy = ax

        min1 = 1e18
        max1 = -1e18
        for v in poly1:
            d = v[0]*axx + v[1]*axy
            if d < min1: min1 = d
            if d > max1: max1 = d

        min2 = 1e18
        max2 = -1e18
        for v in poly2:
            d = v[0]*axx + v[1]*axy
            if d < min2: min2 = d
            if d > max2: max2 = d

        overlap = min(max1, max2) - max(min1, min2)
        if overlap <= 0:
            return 0.0  # no collision
        if overlap < min_overlap:
            min_overlap = overlap

    return min_overlap

@njit(cache=True)
def objective(vals, S):
    penalty = 0.0

    polys = np.empty((N, nsi, 2))
    axes = np.empty((N, nsi, 2))

    for i in range(N):
        x, y, a = vals[i*3:i*3+3]
        polys[i] = transform(x, y, a, unit_poly)
        axes[i] = rotate(a, unit_poly_normals)
        penalty += container_penalty(polys[i], S)

    for i in range(N):
        for j in range(i+1, N):
            combined_axes = np.vstack((axes[i], axes[j]))
            overlap = sat_overlap(polys[i], polys[j], combined_axes)
            if overlap > 0:
                penalty += overlap * overlap

    return penalty

def run_attempt(seed):
    np.random.seed(seed)

    S = np.sqrt(N) * (2 + np.random.rand()*2)
    initial_S = S

    x0 = np.random.uniform(-S/2, S/2, N*3)

    best_x = x0.copy()
    best_S = S

    while True:
        res = minimize(objective, x0, args=(S,), method="L-BFGS-B")

        shrink = 1 - final_step - (S - np.sqrt(N)*nsi/nsc) * (0.01 - final_step) / (initial_S - np.sqrt(N)*nsi/nsc)

        if res.fun < penalty_tol:
            best_x, best_S = res.x.copy(), S
            x0 = res.x * shrink
            S *= shrink
        else:
            bh = basinhopping(objective, x0,
                              minimizer_kwargs={"args": (S,)},
                              niter=50)
            if bh.fun < penalty_tol:
                best_x, best_S = bh.x.copy(), S
                x0 = bh.x * shrink
                S *= shrink
            else:
                break

    return best_S, best_x

results = Parallel(n_jobs=-1)(delayed(run_attempt)(i) for i in range(attempts))
best_S, best_vals = min(results, key=lambda x: x[0])

side_len = best_S * np.sin(np.pi/nsc) / np.sin(np.pi/nsi)
print("Final side length:", side_len)

fig, ax = plt.subplots()

cont = np.vstack((unit_cont * best_S, unit_cont[0]*best_S))
ax.plot(cont[:,0], cont[:,1], 'k-', lw=0.5)

for i in range(N):
    x, y, a = best_vals[i*3:i*3+3]
    poly = transform(x, y, a, unit_poly)
    poly = np.vstack((poly, poly[0]))
    ax.fill(poly[:,0], poly[:,1], "#ccc", edgecolor="black", lw=0.5)

ax.set_aspect("equal")
plt.title(f"Side length: {side_len}")
plt.savefig(f"{N}_{nsi}_in_{nsc}.png")
