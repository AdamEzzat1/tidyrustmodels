"""Exact rational OLS baseline, independent of R and Rust.

Solves the normal equations (X'X) b = X'y in exact arithmetic with fractions.Fraction
for a small integer dataset, then prints the correctly rounded f64 solution plus the
fitted values. Used by crates/tidyrustmodels-models/tests/baselines.rs and by the R
reference harness as System X (see docs/specs/verification-design.md).
"""
from fractions import Fraction as F
import json, sys

# Deterministic small dataset: n = 12, two predictors, intercept. Integer entries so
# every quantity below is an exact rational.
x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
x2 = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11]
y  = [3, 4, 9, 2, 13, 5, 12, 8, 11, 6, 17, 20]

def solve(X, y):
    n, p = len(X), len(X[0])
    XtX = [[sum(F(X[i][a]) * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [sum(F(X[i][a]) * y[i] for i in range(n)) for a in range(p)]
    # Gauss-Jordan in exact arithmetic
    A = [row[:] + [Xty[r]] for r, row in enumerate(XtX)]
    for c in range(p):
        piv = next(r for r in range(c, p) if A[r][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        A[c] = [v / A[c][c] for v in A[c]]
        for r in range(p):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [A[r][p] for r in range(p)]

X = [[1, a, b] for a, b in zip(x1, x2)]
beta = solve(X, y)
fitted = [sum(F(X[i][j]) * beta[j] for j in range(3)) for i in range(len(X))]
out = {
    "x1": x1, "x2": x2, "y": y,
    "beta_exact": [str(b) for b in beta],
    "beta_f64": [float(b) for b in beta],
    "fitted_f64": [float(v) for v in fitted],
}
json.dump(out, sys.stdout, indent=1)
print()
