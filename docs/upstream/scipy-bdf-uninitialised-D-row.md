# Upstream report draft: `scipy.integrate.BDF` reads an uninitialised row of `D`

**Status: NOT YET FILED.** This is a ready-to-submit draft for
https://github.com/scipy/scipy/issues. It is kept in the repo because the finding is what
`dynameta/core/numerics.py::ZeroInitBDF` exists to work around -- if the upstream issue is ever
fixed, that subclass becomes deletable, and whoever deletes it will want this.

Found 2026-08-31 while diagnosing a ~9% flake of `tests/test_soa.py::test_eh_numba_parity` on
the `numba kernels (windows, py3.12)` CI leg (job 99382994280).

---

## Title

`BDF` reads an uninitialised row of its difference array, which can raise a spurious
`invalid value encountered in subtract` under `np.seterr`/warnings-as-errors

## Describe your issue

`scipy.integrate.BDF` allocates its backward-difference array with `np.empty` and initialises
only two of its rows, then reads a third on the first accepted step. The value it reads cannot
affect the solution -- so this is **not** a wrong-answers bug -- but the arithmetic performed on
it can raise a spurious IEEE INVALID exception whenever the recycled heap happens to contain a
**signalling** NaN at that address.

In `scipy/integrate/_ivp/bdf.py` (line numbers from 1.17.1; 1.18.1 is the same code a few lines
lower), `BDF.__init__` does:

```python
D = np.empty((MAX_ORDER + 3, self.n), dtype=self.y.dtype)   # :249
D[0] = self.y                                               # :250
D[1] = f * self.h_abs * self.direction                      # :251
```

`D[2:]` is never written. Then in `_step_impl`, after the first accepted step -- where `order`
is still its initial value of 1 -- the difference update runs:

```python
D[order + 2] = d - D[order + 1]      # :416   ->   D[3] = d - D[2]
```

`D[2]` has not been written by anyone, so this reads `n` doubles of uninitialised memory.

**Why it does not corrupt the solution.** The very next block returns early on the first step:

```python
if self.n_equal_steps < order + 1:   # :421   (1 < 2 on the first step)
    return True, None
```

so the garbage-derived `D[3]` is never consumed by the order-selection heuristic at `:431`, and
step 2 overwrites it from a legitimate `D[order + 1]`. Filling `D[2:]` with `0`, `1`, `-1e300`,
`+inf` or `1e12 * randn` leaves the integration bit-identical, which confirms the value is inert.

**Why it is still a problem.** The subtraction itself executes. IEEE-754 requires
`finite - sNaN` to signal INVALID, and roughly 1 bit pattern in 3000-4000 is a signalling NaN
(exponent all ones, mantissa MSB clear, mantissa nonzero). So whenever the recycled heap block
behind that `np.empty` spells an sNaN in the right slot, NumPy raises

```
RuntimeWarning: invalid value encountered in subtract
```

from a line of SciPy's own arithmetic, attributed to the caller's model. Projects that run
`filterwarnings = ["error"]` (or `np.seterr(invalid='raise')`) see this as a hard, intermittent
failure with a traceback pointing at their ODE right-hand side. We measured it at ~9% of CI runs
(1 in 11) for a solve that builds 14 `BDF` instances in one process.

## Reproducing code example

The read is deterministic; whether it *signals* depends on heap contents, so the reproducer
plants the bit pattern that the heap sometimes supplies:

```python
import struct
import warnings
import numpy as np
from scipy.integrate import BDF

SNAN = struct.unpack('<d', struct.pack('<Q', 0x7FF0000000000001))[0]   # signalling NaN

sol = BDF(lambda t, y: -y, 0.0, np.array([1.0, 2.0]), 10.0)
sol.D[2:] = SNAN                    # what an unlucky np.empty hands you
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    sol.step()
print([str(x.message) for x in w])
# -> ['invalid value encountered in subtract']        (bdf.py:416)
```

Note there is no `np.seterr` call: NumPy's default `invalid` mode is already `'warn'`, so this
reproduces under stock settings.

Two controls worth noting:

* A **quiet** NaN in the same place produces **no** warning -- `finite - qNaN` does not signal.
  So the trigger is specifically an sNaN bit pattern, not "NaN in memory".
* A finite sentinel shows the read plainly: put `-12345.678` in `D[2]`, take one step, and the
  value appears in `D[3]` -- it was consumed before it was ever written.

## Error message

```
RuntimeWarning: invalid value encountered in subtract
  File ".../scipy/integrate/_ivp/bdf.py", line 416, in _step_impl
    D[order + 2] = d - D[order + 1]
```

## Suggested fix

One line -- zero the rows that are not otherwise initialised:

```python
D = np.zeros((MAX_ORDER + 3, self.n), dtype=self.y.dtype)
```

or, keeping `np.empty` for the two rows that are written immediately:

```python
D[2:] = 0.0
```

Zero is not merely a safe filler here: the higher backward differences of a solution known at a
single point genuinely are zero, so this is also the mathematically correct initial content. We
verified the change is bit-identical on real solves (same step count, same `y` to all 17
digits).

`LSODA`/`Radau` were not audited for the same pattern.

## SciPy/NumPy/Python version information

Observed with SciPy 1.17.1 and 1.18.1 (the code is unchanged between them), NumPy 2.x,
CPython 3.12-3.14, Windows and Linux.
