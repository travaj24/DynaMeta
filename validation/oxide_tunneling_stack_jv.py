"""
SELF-CONSISTENT GATE-OXIDE TUNNELLING against MEASURED dielectric-stack J-V data (plan item 4.1.1, gate O13):
A STUB THAT SKIPS -- no published or team SiO2 / high-k stack J-V data is in hand (user decision 16, 2026-11-06),
so the stack path (several dielectric layers, Anderson offsets from fitted barriers, per-layer masses) is
validated ONLY against the kernel's independent quadrature oracles (tests/test_tunneling_kernels.py K6 / K9 /
K10) and the device's D-continuity and split-region identities (tests/test_devsim_oxide_tunneling_sc.py O12).
That is a statement about the NUMERICS, not about whether the WKB / Tsu-Esaki model with fitted barriers
reproduces a real stack.

TO CLOSE THE GAP, drop one JSON file per measured stack into validation/data/oxide_stack_jv/ :

    {"source": "<citation: paper, figure / table number>",
     "semiconductor": {"material": "Si", "doping_m3": -1e23, "n_i_m3": 1e16, "N_dos_m3": 2.8e25,
                       "transverse_mass_over_m0": 0.5},
     "layers": [{"name": "ox_a", "t_m": 1.0e-9, "eps_r": 3.9, "barrier_eV": 3.1, "against": "Si",
                 "m_over_m0": 0.42}, ...],                      # semiconductor side first
     "gate": {"work_function_eV": 4.6, "fermi_energy_eV": 8.0},
     "chi_eV": {"Si": 4.05},
     "points": [[V_gate_V, J_A_m2], ...],
     "rel_uncertainty": 0.3}                                    # the paper's quoted J uncertainty

and implement the comparison below: build the 1-D MOS with tests/_oxide_tunneling_child.Dev (its `layers`
argument already takes any stack; the gate offset from tunneling.gate_offset_for_work_function), install
setup_oxide_tunneling with one DielectricBarrier per layer, sweep the points and gate |J/J_meas - 1| against
rel_uncertainty. Until a file exists this script reports SKIP (exit 42), which the full tier counts
separately from a pass.

Run:  python -m validation.oxide_tunneling_stack_jv
"""
import glob
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from validation._harness import Harness                                 # noqa: E402

DATA = os.path.join(HERE, "validation", "data", "oxide_stack_jv")


def main():
    H = Harness("oxide_tunneling_stack_jv")
    H.require(importlib.util.find_spec("devsim") is not None, "devsim not installed")
    files = sorted(glob.glob(os.path.join(DATA, "*.json")))
    if not files:
        H.skip("O13 measured stack J-V", "needs measured J-V: no published or team SiO2 / high-k stack data in "
               "hand (user decision 16); add validation/data/oxide_stack_jv/*.json (format in the module "
               "docstring)")
        return H.summary()
    for f in files:
        H.gate("O13 " + os.path.basename(f), False,
               "data present but the comparison is not implemented yet (module docstring: TO CLOSE THE GAP)")
    return H.summary()


if __name__ == "__main__":
    raise SystemExit(main())
