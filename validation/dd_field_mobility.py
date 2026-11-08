"""Validate field-dependent mobility mu(E) (Caughey-Thomas velocity saturation) in the DEVSIM
drift-diffusion solve (roadmap R1). An ITO bar with ohmic ends: in a uniform resistor the field is
E = V/L, so the drift current is I = q N v(E) A with v(E) = mu_low E / (1+(mu_low E/v_sat)^b)^(1/b) --
ohmic at low bias, saturating to q N v_sat A at high bias. Independent reference = that analytic v(E).

GATE A (low-bias Ohm): mu_low E << v_sat -> I = V sigma A/L, sigma = q N mu_low; rel-diff < 1e-6 (the
        Newton stop, rel_tol 1e-6; MEASURED 5.0e-16).
GATE B (high-bias saturation): at 4 V (mu_low E/v_sat = 1.6) the current equals q N v(E) A of the uniform
        field E = V/L (rel < 1e-6; MEASURED 0.0) AND the J-V is concave (chord G 53 % of ohmic, below the
        ohmic extrapolation) -- the saturation signature.
GATE C (reduces to constant): field_mobility with a huge v_sat (mu -> mu_low everywhere) reproduces the
        constant-mobility (field_mobility=False) contact current to ~1e-6 -- the off-switch proof.

MESH (2026-11-04). The bar was an unstructured gmsh mesh, on which DEVSIM sums the MAGNITUDES of the
circumcentric dual pieces (TetrahedronEdgeCouple.cc:99), so its couplings / node volumes were inflated
(MEASURED sum NodeVolume = 1.151 x the bar volume) and the low-bias current read 14.2 % HIGH -- which
gate A's old "< 0.16, mesh-limited" tolerance absorbed: it was testing Ohm's law only to 16 % and calling
the DEVSIM volume defect discretization error. Gate B could not match q N v(E) A either (I_hi/(qN v_sat A)
read 1.16 against the uniform-field 0.848), which its comment attributed to carrier redistribution under
the strong field; on the exact mesh the uniform bar stays uniform and the match is exact, so gate B now
checks it. Gate C (a RATIO of two solves on the same mesh) passed because the defect cancels in it. The
bar is now devsim_3d.kuhn_bar_mesh (exact Kuhn-tensor tets, in memory, no fixed file).

Run: python -m validation.dd_field_mobility
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynameta.carriers.devsim_env import import_devsim   # D0: the ONE devsim import home (BLAS pin)

ds = import_devsim()


from dynameta.carriers import physics_drift_diffusion as DD
from dynameta.carriers.dc_solve import solve_dc
from dynameta.carriers.physics_equilibrium import M_E
from dynameta.carriers.mobility import drift_velocity

LX, LY, LZ = 100.0, 80.0, 80.0          # nm (short bar -> high field at modest bias)
N_BG, MU, EPS = 4e26, 0.004, 9.5        # ITO-like
V_SAT = 1.0e5                            # m/s
BETA = 2.0
Q = 1.602176634e-19
A = (LY * 1e-9) * (LZ * 1e-9)
L = LX * 1e-9


def _build(mesh, dev):
    from dynameta.carriers import devsim_3d as D3
    D3.kuhn_bar_mesh(mesh, (LX, LY, LZ), 8.0, region="semi", contacts=("left", "right"))   # exact, in memory
    ds.add_gmsh_region(mesh=mesh, gmsh_name="semi", region="semi", material="ITO")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="left", region="semi", name="left", material="metal")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="right", region="semi", name="right", material="metal")
    ds.finalize_mesh(mesh=mesh); ds.create_device(mesh=mesh, device=dev)
    return dev


def _current_at(dev, V, abs_tol, n_steps):
    ds.set_parameter(device=dev, name="left_bias", value=0.0)
    ds.set_parameter(device=dev, name="right_bias", value=0.0)
    solve_dc(dev, method="newton", abs_tol=abs_tol, rel_tol=1e-6, max_iter=120,
             semiconductor_regions=["semi"])
    for k in range(1, n_steps + 1):
        ds.set_parameter(device=dev, name="right_bias", value=V * k / n_steps)
        solve_dc(dev, method="newton", abs_tol=abs_tol, rel_tol=1e-6, max_iter=120,
                 semiconductor_regions=["semi"])
    return abs(ds.get_contact_current(device=dev, contact="right",
                                      equation="ElectronContinuityEquation"))


def _setup(dev, *, field_mobility, v_sat):
    DD.setup_semiconductor_region_dd(dev, "semi", n_bg_m3=N_BG, eps_static=EPS,
                                     dos_mass_kg=0.35 * M_E, mobility_m2Vs=MU,
                                     field_mobility=field_mobility, v_sat_ms=v_sat, ct_beta=BETA)
    for c in ("left", "right"):
        DD.setup_contact_ohmic_dd(dev, c)


def _I_ref(V):
    return Q * N_BG * drift_velocity(V / L, MU, V_SAT, BETA) * A    # q N v(E) A, E = V/L


def main():
    print("[fm] === DD field-dependent mobility (Caughey-Thomas velocity saturation) ===", flush=True)
    abs_tol = max(1e10, N_BG * 1e-12)

    # GATE A + B: field mobility ON, low and high bias vs the analytic v(E) reference
    dev = _build("m_on", "dev_on")
    _setup(dev, field_mobility=True, v_sat=V_SAT)
    I_lo = _current_at(dev, 0.01, abs_tol, 2)          # E=1e5 V/m: mu_low*E/vsat=0.004 -> ohmic
    relA = abs(I_lo - _I_ref(0.01)) / _I_ref(0.01)
    g_a = relA < 1e-6
    print("[fm] A low-bias Ohm: I={:.3e} A ref={:.3e} rel={:.2e} -> {}".format(
        I_lo, _I_ref(0.01), relA, "OK" if g_a else "FAIL"), flush=True)

    # GATE B: the uniform-field q*N*v(E)*A at high bias AND the velocity-saturation SIGNATURE (a CONCAVE
    # J-V: the chord conductance G=I/V drops with bias, the high-bias current falls below the ohmic
    # extrapolation). On the gmsh mesh the match was impossible (I_hi/(qN v_sat A) = 1.16, the inflated
    # couplings) and the old comment blamed carrier redistribution; on the exact Kuhn mesh the uniform bar
    # stays uniform and I_hi equals q N v(E) A (MEASURED rel 0.0; module docstring, MESH).
    I_hi = _current_at(dev, 4.0, abs_tol, 24)          # E=4e7 V/m: mu_low*E/v_sat=1.6 -> v ~ 0.85 v_sat
    G_lo, G_hi = I_lo / 0.01, I_hi / 4.0
    I_lin = G_lo * 4.0                                  # constant-mobility ohmic extrapolation at 4 V
    I_sat = Q * N_BG * V_SAT * A
    relB = abs(I_hi - _I_ref(4.0)) / _I_ref(4.0)
    g_b = (G_hi < 0.85 * G_lo) and (I_hi < 0.9 * I_lin) and (relB < 1e-6)
    print("[fm] B saturation signature: G(V) {:.3e}->{:.3e} S ({:.0%} of ohmic), I_hi={:.3e} < ohmic "
          "extrap {:.3e}; I_hi/(qN v_sat A)={:.4f}, vs q N v(E) A rel={:.2e} -> {}".format(
              G_lo, G_hi, G_hi / G_lo, I_hi, I_lin, I_hi / I_sat, relB, "OK" if g_b else "FAIL"), flush=True)

    # GATE C: reduces to the constant-mobility solve (huge v_sat -> mu == mu_low) to ~1e-6
    dev_off = _build("m_off", "dev_off")
    _setup(dev_off, field_mobility=False, v_sat=None)
    I_off = _current_at(dev_off, 0.5, abs_tol, 6)
    dev_big = _build("m_big", "dev_big")
    _setup(dev_big, field_mobility=True, v_sat=1.0e12)   # v_sat huge -> velocity factor -> 1
    I_big = _current_at(dev_big, 0.5, abs_tol, 6)
    relC = abs(I_big - I_off) / abs(I_off)
    g_c = relC < 1e-5
    print("[fm] C reduces-to-constant: I(field,vsat=1e12)={:.6e}  I(constant)={:.6e}  rel={:.2e} -> {}".format(
        I_big, I_off, relC, "OK" if g_c else "FAIL"), flush=True)

    ok = g_a and g_b and g_c
    print("[fm] *** DD FIELD MOBILITY: {} ***".format("PASS" if ok else "FAIL"), flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
