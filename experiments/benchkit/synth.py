"""The cost benchmark's synthetic grid: class-structured series whose size
is set one axis at a time around a base point, so fit/predict time and
memory can be read as functions of n, m and C separately. Every arm sees identical arrays: the generator is seeded by
the cell's size.
"""

import re

import numpy as np

BASE = dict(n_train=1000, m=256, C=1)
AXES = {"n_train": (250, 1000, 4000, 16000, 64000),
        "m": (64, 256, 1000, 4000, 16000),
        "C": (1, 4, 16, 64, 256)}
K = 5
_NAME = re.compile(r"^syn_n(\d+)_m(\d+)_c(\d+)$")


def cell_name(n_train, m, C):
    return f"syn_n{n_train}_m{m}_c{C}"


def parse_cell(name):
    match = _NAME.match(name)
    if not match:
        raise ValueError(f"not a cost-grid cell: {name!r}")
    n, m, c = (int(v) for v in match.groups())
    return dict(n_train=n, m=m, C=c)


def grid():
    """13 cell names: the base point, then each axis' other rungs ascending."""
    names = [cell_name(**BASE)]
    for axis, rungs in AXES.items():
        for rung in rungs:
            point = dict(BASE, **{axis: rung})
            name = cell_name(**point)
            if name not in names:
                names.append(name)
    return names


def axes_of(name):
    """{axis: rung} for every axis this cell sits on (the base point sits on
    all three; any other cell on exactly one)."""
    point = parse_cell(name)
    out = {}
    for axis in AXES:
        if all(point[o] == BASE[o] for o in AXES if o != axis):
            out[axis] = point[axis]
    return out


def _seed(n_train, m, C, seed):
    return (n_train * 1_000_003 + m * 1_009 + C * 7 + seed) % (2 ** 32)


def synthetic_cost(n_train, m, C, seed=0):
    """(X_tr, y_tr, X_ts, y_ts), float64 (n, C, m), labels 0..K-1 balanced,
    n_test = n_train // 2. Per class and channel a template of three
    sinusoids plus one Gaussian bump; per series an amplitude factor, a
    circular shift and white noise at 20 % of the templates' std."""
    rng = np.random.RandomState(_seed(n_train, m, C, seed))
    t = np.arange(m) / m
    idx = np.arange(m)
    templates = np.empty((K, C, m))
    for k in range(K):
        for c in range(C):
            tpl = np.zeros(m)
            for _ in range(3):
                f = np.exp(rng.uniform(np.log(0.5), np.log(8.0)))
                tpl += rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * f * t
                                                      + rng.uniform(0, 2 * np.pi))
            centre = rng.uniform(0.15, 0.85) * m
            tpl += rng.choice([-2.0, 2.0]) * np.exp(-0.5 * ((idx - centre) / (m / 16)) ** 2)
            templates[k, c] = tpl
    noise = 0.2 * float(templates.std())

    def make(n):
        y = (np.arange(n) % K).astype(np.int64)
        X = templates[y] * rng.uniform(0.8, 1.2, size=(n, 1, 1))
        shifts = rng.randint(-(m // 10), m // 10 + 1, size=n)
        for i in range(n):
            if shifts[i]:
                X[i] = np.roll(X[i], shifts[i], axis=-1)
        X += rng.randn(n, C, m) * noise
        return np.ascontiguousarray(X), y
    X_tr, y_tr = make(n_train)
    X_ts, y_ts = make(n_train // 2)
    return X_tr, y_tr, X_ts, y_ts
