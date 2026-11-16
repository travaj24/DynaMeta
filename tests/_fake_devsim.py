"""Test helper (not collected): a small RECORDING stand-in for the part of the devsim API that the carrier
setup / diagnostic helpers call, plus a restricted evaluator for the model strings they emit.

Why: the solver-free CI legs have no devsim, yet several v0.16 carrier helpers are plain numpy around a few
devsim reads and writes (parameters, node values, model strings). With this stand-in the tests drive them
on hand-made node data and check the RESULT against an independent oracle -- a hand-written derivative
against a central finite difference of the emitted value expression, a closed-form limit, a refusal naming
its cause. Nothing here solves anything: the stand-in stores what it is given and evaluates model strings
only on request.

Isolation: install() patches the ONE devsim import home (devsim_env.import_devsim) and the registry's lazy
shim (eq_registry._DevsimShim._mod) through pytest's monkeypatch, so both are restored after the test and no
module keeps the stand-in. modules_imported_with() imports a module whose top level calls import_devsim()
(physics_equilibrium_dg, physics_density_gradient) on a leg without devsim, and removes every module it
bound to a stand-in (and its attribute on the parent package) afterwards, so later tests still see the real
ImportError.
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import re
import sys

import numpy as np

# a model-string token: an identifier, optionally "<model>:<variable>" (a derivative model) and "@n0"/"@n1"
# (an edge end); not preceded by a digit or a dot, so the exponent of 1e-9 is not read as a name
_TOKEN = re.compile(r"(?<![0-9.])[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)?(?:@n[01])?")
_FUNCS = {"exp": np.exp, "log": np.log, "abs": np.abs, "pow": np.power,
          "min": np.minimum, "max": np.maximum}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}


def evaluate(expr: str, values) -> np.ndarray:
    """Evaluate a devsim-style model string ('^' is the power, names may carry ':' and '@n0') with numpy on
    `values` ({name: number or array}). Only + - * / ^, unary minus, numeric literals, names and the
    functions exp, log, abs, pow, min, max are accepted; anything else raises (so an unexpected construct in
    an emitted string fails the test instead of being guessed)."""
    names = {}

    def sub(m):
        tok = m.group(0)
        if tok in _FUNCS:
            return tok
        return names.setdefault(tok, "_v{}".format(len(names)))

    src = _TOKEN.sub(sub, expr.replace("^", "**"))
    inv = {v: k for k, v in names.items()}
    tree = ast.parse(src, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            key = inv[node.id]
            if key not in values:
                raise KeyError("model string names {!r}, which the test did not supply".format(key))
            return np.asarray(values[key], dtype=np.float64)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("unsupported construct in model string: {}".format(ast.dump(node)))

    return ev(tree)


class FakeDevsim:
    """Recording stand-in for the devsim calls the tested helpers make. Regions have a node count; node
    values are stored arrays or, for a node model created with node_model(), its equation evaluated on the
    region's stored arrays and parameters (region parameters first, then device parameters)."""

    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        self.n_nodes = {}            # region -> node count
        self.params = {}             # (region or None, name) -> float
        self.nodes = {}              # (region, name) -> array
        self.node_eqs = {}           # (region, name) -> model string
        self.edge_eqs = {}           # (region, name) -> model string
        self.edge_from_node = []     # (region, node model)
        self.contact_eqs = {}        # (contact, name) -> model string
        self.equations = []          # kwargs of ds.equation
        self.contact_equations = []  # kwargs of ds.contact_equation
        self.interfaces = {}         # interface -> (region0, region1)
        self.contacts = {}           # contact -> region
        self.materials = {}          # region -> material
        self.region_equations = {}   # region -> [equation names]
        self.numbers = {}            # (region, "equation" | "variable", name) -> global index array
        self.deleted = []            # names passed to delete_node_model
        self.writes = 0              # every mutating call (a refusal must make none)

    # ---- topology -------------------------------------------------------------------------------
    def add_region(self, region, n, material="mat"):
        self.n_nodes[region] = int(n)
        self.materials[region] = material

    def get_dimension(self, device):
        return self.dimension

    def get_region_list(self, device, interface=None, contact=None):
        if interface is not None:
            return list(self.interfaces[interface])
        if contact is not None:
            return [self.contacts[contact]]
        return list(self.n_nodes)

    def get_interface_list(self, device):
        return list(self.interfaces)

    def get_contact_list(self, device):
        return list(self.contacts)

    def get_material(self, device, region):
        return self.materials[region]

    def get_equation_list(self, device, region):
        return list(self.region_equations.get(region, []))

    def get_equation_numbers(self, device, region, equation=None, variable=None):
        key = (region, "equation", equation) if equation is not None else (region, "variable", variable)
        return list(self.numbers[key])

    # ---- parameters -------------------------------------------------------------------------------
    def get_parameter(self, device, name, region=None):
        if region is not None and (region, name) in self.params:
            return self.params[(region, name)]
        return self.params[(None, name)]

    def set_parameter(self, device=None, name=None, value=None, region=None):
        self.writes += 1
        self.params[(region, name)] = float(value)

    def get_parameter_list(self, device, region=None):
        return [n for (r, n) in self.params if r == region]

    # ---- node data and models ----------------------------------------------------------------------
    def _values(self, region):
        out = {n: v for (r, n), v in self.params.items() if r is None}
        out.update({n: v for (r, n), v in self.params.items() if r == region})
        out.update({n: v for (r, n), v in self.nodes.items() if r == region})
        return out

    def get_node_model_values(self, device, region, name):
        if (region, name) in self.nodes:
            return list(self.nodes[(region, name)])
        if (region, name) in self.node_eqs:
            v = evaluate(self.node_eqs[(region, name)], self._values(region))
            return list(np.broadcast_to(v, (self.n_nodes[region],)).astype(float))
        raise KeyError("no node model {!r} on region {!r}".format(name, region))

    def set_node_values(self, device, region, name, values):
        self.writes += 1
        self.nodes[(region, name)] = np.asarray(values, dtype=np.float64).copy()

    def node_solution(self, device, region, name):
        self.writes += 1
        self.nodes.setdefault((region, name), np.zeros(self.n_nodes[region]))

    def node_model(self, device, region, name, equation):
        self.writes += 1
        self.node_eqs[(region, name)] = equation

    def delete_node_model(self, device, region, name):
        self.writes += 1
        self.node_eqs.pop((region, name), None)
        self.deleted.append(name)

    def get_node_model_list(self, device, region):
        return sorted({n for (r, n) in self.nodes if r == region} | {n for (r, n) in self.node_eqs if r == region})

    def edge_from_node_model(self, device, region, node_model):
        self.writes += 1
        self.edge_from_node.append((region, node_model))

    def edge_model(self, device, region, name, equation):
        self.writes += 1
        self.edge_eqs[(region, name)] = equation

    def get_edge_model_list(self, device, region):
        return sorted(n for (r, n) in self.edge_eqs if r == region)

    def contact_node_model(self, device, contact, name, equation):
        self.writes += 1
        self.contact_eqs[(contact, name)] = equation

    def equation(self, **kw):
        self.writes += 1
        self.equations.append(kw)

    def contact_equation(self, **kw):
        self.writes += 1
        self.contact_equations.append(kw)


def install(monkeypatch, fake: FakeDevsim) -> FakeDevsim:
    """Route every devsim access of the carrier package to `fake` for one test (restored by monkeypatch)."""
    from dynameta.carriers import devsim_env
    from dynameta.carriers import eq_registry as R
    monkeypatch.setattr(devsim_env, "import_devsim", lambda: fake)
    monkeypatch.setattr(R._DevsimShim, "_mod", fake)
    return fake


@contextlib.contextmanager
def modules_imported_with(*names):
    """Yield the modules `names` (dotted), importing them with a stand-in devsim when the real one is
    missing; on exit remove every module this call bound to a stand-in (and its parent-package attribute),
    so the rest of the session is unaffected. With devsim installed the real modules are yielded untouched."""
    from dynameta.carriers import devsim_env
    before = set(sys.modules)
    try:
        mods = [importlib.import_module(n) for n in names]
        added = set()
    except ImportError:
        orig = devsim_env.import_devsim
        devsim_env.import_devsim = lambda: FakeDevsim()
        try:
            mods = [importlib.import_module(n) for n in names]
        finally:
            devsim_env.import_devsim = orig
        # only the modules bound to a stand-in at import are dropped afterwards; devsim-free modules imported
        # on the way stay (dropping them would give later importers second copies of their classes)
        added = {n for n in set(sys.modules) - before
                 if isinstance(getattr(sys.modules[n], "ds", None), FakeDevsim)}
    try:
        yield mods
    finally:
        for n in added:
            sys.modules.pop(n, None)
            parent, _, child = n.rpartition(".")
            pkg = sys.modules.get(parent)
            if pkg is not None and child in vars(pkg):
                delattr(pkg, child)
