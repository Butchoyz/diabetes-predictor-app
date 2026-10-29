"""
Exact Shapley values computed straight from the thesis formulas, from every coalition of players.

A "player" is one input of the function being explained: a clinical feature (8 players) for the 9
models, or a base learner's prediction (7 players) for the Stacking Ensemble's meta-learner.

* f(S)  - coalition value: the average output over the reference patients when the players in S
          take this patient's values and the others keep each reference patient's own values.
          f(∅) is the base value ϕ0 (Formula 8: the average prediction over the reference
          patients) and f(F) is the prediction for this patient.
* ϕi    - Formula 7:  ϕi = Σ_{S ⊆ F\\{i}} |S|!(|F|-|S|-1)!/|F|! · [f(S ∪ {i}) - f(S)]
* g(z') - Formula 6:  g(z') = ϕ0 + Σ ϕi z'i, which equals f(F) (local accuracy).

This is the same computation as ``shap.explainers.Exact`` with an ``Independent`` masker over the
same reference patients (checked by ``verify_shap.py``). It is done by hand here so that every
intermediate number of Formula 7 can be shown in the app.
"""
from fractions import Fraction
from math import factorial

import numpy as np


def coalition_masks(n_players: int) -> np.ndarray:
    """Every subset S of the players as a boolean row: row s contains player j iff bit j of s is set."""
    return ((np.arange(2 ** n_players)[:, None] >> np.arange(n_players)) & 1).astype(bool)


def count_players(v: np.ndarray) -> int:
    return int(round(np.log2(len(v))))


def coalition_values(predict_fn, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    """f(S) for all 2^M subsets S, indexed by bitmask (see ``coalition_masks``).

    ``predict_fn`` maps an (n, M) array to n outputs; ``background`` holds the reference rows.
    """
    masks = coalition_masks(len(x))
    rows = np.where(masks[:, None, :], x[None, None, :], background[None, :, :])
    preds = np.asarray(predict_fn(rows.reshape(-1, len(x))), dtype=float)
    return preds.reshape(len(masks), len(background)).mean(axis=1)


def subset_weight(size: int, n_players: int) -> Fraction:
    """Formula 7's weight of one subset S: |S|!(|F|-|S|-1)!/|F|!"""
    return Fraction(factorial(size) * factorial(n_players - size - 1), factorial(n_players))


def _subsets_without(n_players: int, i: int):
    """Bitmasks of all subsets S ⊆ F\\{i}, and their sizes |S|."""
    masks = coalition_masks(n_players)
    subsets = np.flatnonzero(~masks[:, i])
    return subsets, masks[subsets].sum(axis=1)


def shapley_values(v: np.ndarray) -> np.ndarray:
    """Formula 7 for every player, from the coalition values v."""
    n = count_players(v)
    phi = np.zeros(n)
    for i in range(n):
        subsets, sizes = _subsets_without(n, i)
        weights = np.array([float(subset_weight(k, n)) for k in sizes])
        phi[i] = np.sum(weights * (v[subsets | (1 << i)] - v[subsets]))
    return phi


def shapley_breakdown(v: np.ndarray, i: int) -> list:
    """Formula 7 for player i grouped by subset size |S|, the layout of the thesis' worked example.

    Per group: the number of subsets, the weight of one subset, the average marginal contribution
    f(S ∪ {i}) - f(S) over those subsets, and the group total = count × weight × average.
    The group totals add up to ϕi.
    """
    n = count_players(v)
    subsets, sizes = _subsets_without(n, i)
    groups = []
    for k in range(n):
        s = subsets[sizes == k]
        marginal = v[s | (1 << i)] - v[s]
        weight = subset_weight(k, n)
        groups.append({
            "size": k,
            "count": len(s),
            "weight": weight,
            "f_S": float(v[s[0]]) if len(s) == 1 else None,          # shown when the group is one subset
            "f_S_i": float(v[s[0] | (1 << i)]) if len(s) == 1 else None,
            "mean_marginal": float(marginal.mean()),
            "total": len(s) * float(weight) * float(marginal.mean()),
        })
    return groups


def subset_contributions(v: np.ndarray, i: int, size: int) -> list:
    """Every subset S ⊆ F\\{i} of one size, with f(S), f(S ∪ {i}) and its weighted contribution."""
    n = count_players(v)
    subsets, sizes = _subsets_without(n, i)
    weight = float(subset_weight(size, n))
    rows = []
    for s in subsets[sizes == size]:
        f_s, f_s_i = float(v[s]), float(v[s | (1 << i)])
        rows.append({
            "members": [j for j in range(n) if s >> j & 1],
            "f_S": f_s,
            "f_S_i": f_s_i,
            "marginal": f_s_i - f_s,
            "weighted": weight * (f_s_i - f_s),
        })
    return rows
