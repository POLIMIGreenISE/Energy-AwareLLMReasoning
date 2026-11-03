"""Refining a policy family's parameter grid at a constraint boundary.

Every matched-energy comparison in this project reads each policy family at its best operating point
subject to energy <= cap (and, in the SLO experiments, p95 <= target). Each family is swept over a
parameter grid built in score space and mapped through that family's own price. Dense in score space
is not dense in ENERGY, and a family whose grid has no point just below the cap is then scored at
whatever point it does have -- which charges it for budget its grid never offered rather than for how
it allocates.

This was not hypothetical. With the published grids, the load-aware family's operating points on the
saturated regime jumped from 23.2 Wh to 26.4 Wh across a 26.1 Wh cap, so 11% of the budget was
unreachable and its published gap at that cap (-3.25 pp) overstated the loss by about half. The
static family's grid, being the score grid itself, lands on almost any cap, so only one side of the
comparison was affected. See results/CORRECTIONS.md.

refine_to_bound() closes that hole wherever it appears, by bisecting between the two operating points
that straddle the boundary and evaluating a few neighbours. Applying it to every family keeps the
comparison about allocation rather than about the shape of each family's price map.
"""


def _mid(a, b, scale):
    """Midpoint between two parameters; None when no distinct value exists."""
    if scale == "log" and a > 0 and b > 0:
        m = (a * b) ** 0.5
    elif scale == "int":
        m = round((a + b) / 2.0)
        if m in (a, b):
            return None
    else:
        m = (a + b) / 2.0
    return None if m in (a, b) else m


def _scaled(p, f, scale):
    v = round(p * f) if scale == "int" else p * f
    return None if v == p else v


def refine_to_bound(points, bound, evaluate, metric="energy_wh", scale="log", steps=12,
                    neighbours=(0.98, 0.99, 0.995, 1.005, 1.01, 1.02)):
    """Add operating points either side of `bound` to `points`, in place, and return it.

    points    list of dicts, each carrying "param" and per-trace metric arrays
    bound     the constraint value (a cap in Wh, or a latency target in seconds)
    evaluate  param -> dict of per-trace metric arrays; "param" is added here
    metric    which mean the constraint applies to
    scale     "log" for multiplicative parameters (lambda), "linear" for bounded ones (tau, rate),
              "int" for counts (the oracle's query budget)

    Does nothing when the grid already straddles the boundary tightly, or when every point sits on
    one side of it (a family that cannot reach the cap at all is a real finding, not a grid artefact).
    """
    def mean_of(p):
        return float(p[metric].mean())

    feasible = [p for p in points if mean_of(p) <= bound + 1e-9]
    over = [p for p in points if mean_of(p) > bound + 1e-9]
    if not feasible or not over:
        return points

    good = max(feasible, key=mean_of)["param"]        # closest from below
    bad = min(over, key=mean_of)["param"]             # closest from above
    seen = {round(float(p["param"]), 12) for p in points}

    def add(v):
        key = round(float(v), 12)
        if key in seen:
            return None
        seen.add(key)
        rec = dict(param=float(v), **evaluate(v))
        points.append(rec)
        return rec

    for _ in range(steps):
        mid = _mid(good, bad, scale)
        if mid is None:
            break
        rec = add(mid)
        if rec is None:
            break
        if mean_of(rec) > bound:
            bad = mid
        else:
            good = mid

    for f in neighbours:
        v = _scaled(good, f, scale)
        if v is not None:
            add(v)
    return points
