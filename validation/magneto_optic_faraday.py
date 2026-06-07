"""
Magneto-optic (gyrotropic) Faraday-rotation oracle -- the magneto-optic row of the modulation-
mechanism landscape. Validates core.effects.MagnetoOpticModel (the gyrotropic permittivity tensor
eps = [[eps_r, i g, 0], [-i g, eps_r, 0], [0, 0, eps_r]]) against an INDEPENDENT circular-eigenmode
transfer-matrix reference -- both as a pure-numpy analytic check AND end-to-end through the FEM.

The gyrotropic tensor is genuinely off-diagonal (imaginary, NON-symmetric) -- the hardest anisotropic
case. The off-diagonal FEM solve is now SUPPORTED (GATE D): the earlier failure was mesh.SetPML's
coordinate stretch being wrong for an anisotropic medium, not an NGSolve assembly defect, and
solve_fem now uses an explicit UPML for tensor eps.

PHYSICS: for z-propagation the two normal modes are circular polarizations with n_pm =
sqrt(eps_r +/- g). A linearly polarized wave through a slab of thickness L rotates its plane of
polarization by the Faraday angle theta_F = (pi L / lambda) Re(n_+ - n_-). Each circular mode is solved
as an isotropic slab (vacuum | n_pm | vacuum, Airy) and recombined into the transmitted Jones vector.

GATE A: the model tensor's eigenvalues are {eps_r - g, eps_r, eps_r + g} (the two circular modes + the
        axial mode).
GATE B: the full circular-TMM transmitted-polarization rotation == the analytic bulk theta_F within
        5% (relative) over a range of gyration g (and g = 0 gives no rotation). The small residual is
        the Fabry-Perot / interface correction the bulk theta_F omits but the exact TMM includes.
GATE C: real eps_r + real g -> the tensor is HERMITIAN (eps == eps^H) = an INDEPENDENT proof of
        losslessness that also catches an off-diagonal SIGN error (ig/-ig vs ig/+ig) the real-n_pm
        circular-TMM cannot; a complex-eps_r (absorbing) variant must FAIL Hermiticity. (NOT the
        circular-TMM R+T=1, which a lossless TMM gives for free -- the lossless trap.)
GATE D: the FEM (UPML off-diagonal tensor solve) transmitted power matches the circular-eigenmode
        Jones-TMM -- the fit-independent Poynting flux T_total == 0.5(|t_+|^2+|t_-|^2), the
        single-projection (co-polarized) lstsq T == the co-pol reference, and the Hermitian tensor is
        lossless (R_flux + T_flux = 1, A_independent ~ 0; the energy is rotated into cross-pol, not
        absorbed).

Run: python -m validation.magneto_optic_faraday
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynameta.core.effects import MagnetoOpticModel

LAM = 1550e-9
EPS_R = 2.25
L = 5.0e-6
G_LIST = [0.0, 0.02, 0.05, 0.08]
ROT_RTOL = 0.05            # bulk theta_F omits the (~4%) Fabry-Perot/interface correction the TMM has
ROT_ABS_DEG = 0.05         # absolute floor (the g=0 zero-rotation case)
# GATE D (FEM) parameters -- a short slab + strong gyration for a clear, fast cross-pol signal.
EPS_R_FEM, G_FEM, L_FEM_NM = 4.0, 0.4, 1000.0
TOL_FEM_T = 3e-2           # FEM vs circular-eigenmode Jones-TMM
TOL_FEM_E = 2e-2           # lossless energy closure (flux R+T = 1)


def _slab_rt(n, k0, L):
    """Amplitude (r, t) of vacuum | n | vacuum at normal incidence (single-slab Airy)."""
    beta = n * k0 * L
    r1 = (1.0 - n) / (1.0 + n)          # vacuum -> slab
    t1 = 2.0 / (1.0 + n)
    r2 = (n - 1.0) / (n + 1.0)          # slab -> vacuum
    t2 = 2.0 * n / (n + 1.0)
    ph = np.exp(1j * beta)
    den = 1.0 + r1 * r2 * ph * ph
    return (r1 + r2 * ph * ph) / den, (t1 * t2 * ph) / den


def _faraday(eps_r, g, k0, L):
    """Full circular-TMM: incident x-pol through the gyrotropic slab -> (rotation_deg, R, T)."""
    n_p, n_m = np.sqrt(eps_r + g), np.sqrt(eps_r - g)
    rp, tp = _slab_rt(n_p, k0, L)
    rm, tm = _slab_rt(n_m, k0, L)
    # incident x = (e+ + e-)/sqrt2, e_pm = (x +/- i y)/sqrt2; transmitted/reflected x,y Jones:
    Ex, Ey = 0.5 * (tp + tm), 0.5j * (tp - tm)
    Rx, Ry = 0.5 * (rp + rm), 0.5j * (rp - rm)
    # major-axis orientation of the transmitted ellipse (polarization rotation from x)
    rot = 0.5 * np.arctan2(2.0 * np.real(np.conj(Ex) * Ey), abs(Ex) ** 2 - abs(Ey) ** 2)
    T = abs(Ex) ** 2 + abs(Ey) ** 2            # n_sub = n_sup = 1 -> T = sum |t|^2 over the two modes
    R = abs(Rx) ** 2 + abs(Ry) ** 2
    return float(np.degrees(rot)), float(R), float(T)


def _gate_d_fem():
    """GATE D: the off-diagonal gyrotropic tensor through the actual UPML FEM, vs the circular-
    eigenmode Jones-TMM reference. Returns (ok, message)."""
    from dynameta.materials import Material, MaterialRegistry, ConstantOptical
    from dynameta.geometry import UnitCell, Stack, Layer, Design
    from dynameta.geometry.specs import OpticalSpec, Mesh3DSpec
    from dynameta.core.eps_field import EpsField
    from dynameta.optics.ngsolve_layered import LayeredOpticalBuilder
    from dynameta.optics.eps_assembler import assemble_eps_cf
    from dynameta.optics.solver import solve_fem

    lam_m, k0 = LAM, 2.0 * np.pi / LAM
    n_p, n_m = np.sqrt(EPS_R_FEM + G_FEM), np.sqrt(EPS_R_FEM - G_FEM)
    _, tp = _slab_rt(n_p, k0, L_FEM_NM * 1e-9)
    _, tm = _slab_rt(n_m, k0, L_FEM_NM * 1e-9)
    # x-pol input: transmitted Ex(co)=0.5(tp+tm), Ey(cross)=0.5j(tp-tm)
    T_total_ref = 0.5 * (abs(tp) ** 2 + abs(tm) ** 2)
    T_co_ref = abs(0.5 * (tp + tm)) ** 2

    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("mo", ConstantOptical(complex(EPS_R_FEM, 0.0))))
    cell = UnitCell.square(300e-9)
    stack = Stack(layers=[Layer("s", L_FEM_NM * 1e-9, "mo")],
                  superstrate_material="air", substrate_material="air")
    m3 = Mesh3DSpec(pml_thk_m=600e-9, superstrate_buffer_m=900e-9, substrate_buffer_m=900e-9,
                    maxh_superstrate_m=45e-9, maxh_substrate_m=45e-9, maxh_background_m=25e-9)
    design = Design(name="mo", unit_cell=cell, stack=stack, electrodes=[], materials=reg, mesh_3d=m3)

    eps_tensor = np.asarray(MagnetoOpticModel(eps_r=EPS_R_FEM, g=G_FEM).eps({}, lam_m), dtype=complex)
    geo = LayeredOpticalBuilder(design).build()
    mats = list(geo.mesh.GetMaterials())
    slab = [r for r in mats if geo.material_by_region[r] == "mo"][0]
    ebr = {rg: EpsField(scalar=complex(1.0, 0.0)) for rg in mats}
    ebr[slab] = EpsField(tensor=eps_tensor)
    opt = OpticalSpec(polarization="x", incidence_angle_deg=0.0, linear_solver="umfpack")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_fem(geo, lam_m, assemble_eps_cf(geo, ebr), opt, order=2,
                        n_super=1.0 + 0j, n_sub=1.0 + 0j)
    d_total = abs(res.T_flux - T_total_ref)
    d_co = abs(res.T - T_co_ref)
    e_flux = abs((res.R_flux + res.T_flux) - 1.0)
    a_ind = abs(res.A_independent) if res.A_independent is not None else 0.0
    ok = d_total < TOL_FEM_T and d_co < TOL_FEM_T and e_flux < TOL_FEM_E and a_ind < TOL_FEM_E
    msg = ("[mo] FEM (e={:.1f} g={:.1f} L={:.0f}nm): flux T_total={:.4f} ref={:.4f} dT={:.1e} | "
           "lstsq T_co={:.4f} ref={:.4f} dT={:.1e} | R_flux+T_flux={:.4f} A_ind={:+.4f}".format(
               EPS_R_FEM, G_FEM, L_FEM_NM, res.T_flux, T_total_ref, d_total,
               res.T, T_co_ref, d_co, res.R_flux + res.T_flux, res.A_independent or 0.0))
    return ok, msg


def main():
    print("[mo] === Magneto-optic Faraday rotation (gyrotropic tensor) vs analytic ===", flush=True)
    k0 = 2.0 * np.pi / LAM

    # GATE A: model tensor eigenvalues
    T3 = np.asarray(MagnetoOpticModel(eps_r=EPS_R, g=0.05).eps({}, LAM))
    eig = np.sort(np.linalg.eigvals(T3).real)
    exp = np.sort([EPS_R - 0.05, EPS_R, EPS_R + 0.05])
    gate_a = bool(np.allclose(eig, exp, atol=1e-12))
    print("[mo] tensor eigenvalues {} vs expected {} : {}".format(
        np.round(eig, 5), np.round(exp, 5), "PASS" if gate_a else "FAIL"), flush=True)

    gate_b = gate_c = True
    for g in G_LIST:
        rot_deg, R, T = _faraday(EPS_R, g, k0, L)
        n_p, n_m = np.sqrt(EPS_R + g), np.sqrt(EPS_R - g)
        theta_an = float(np.degrees(np.pi * L / LAM * (n_p - n_m)))     # analytic bulk Faraday angle
        d_rot = abs(abs(rot_deg) - abs(theta_an))                      # magnitudes (sign convention)
        gate_b = gate_b and (d_rot < ROT_RTOL * abs(theta_an) + ROT_ABS_DEG)
        # GATE C (INDEPENDENT losslessness): for real eps_r,g the gyrotropic tensor must be HERMITIAN
        # -- that is the physical reason the medium is lossless, and it catches an off-diagonal SIGN
        # error (ig/-ig vs ig/+ig) that the real-n_pm circular-TMM cannot. (NOT R+T=1, which any
        # lossless TMM gives for free -- the lossless trap.)
        Tg = np.asarray(MagnetoOpticModel(eps_r=EPS_R, g=g).eps({}, LAM))
        herm_viol = float(np.max(np.abs(Tg - Tg.conj().T)))
        gate_c = gate_c and (herm_viol < 1e-12 * (float(np.max(np.abs(Tg))) + 1e-30))
        print("[mo] g={:.2f}: rot_tmm={:+7.3f} deg  theta_analytic={:+7.3f} deg  |d|={:.3e} deg | "
              "Hermiticity-viol={:.1e}".format(g, rot_deg, theta_an, d_rot, herm_viol), flush=True)
    # counter-check that GATE C has teeth: an ABSORBING variant (complex eps_r) is NON-Hermitian
    Tlossy = np.asarray(MagnetoOpticModel(eps_r=complex(EPS_R, 0.1), g=0.05).eps({}, LAM))
    gate_c = gate_c and (float(np.max(np.abs(Tlossy - Tlossy.conj().T))) > 1e-3)   # must FAIL Hermiticity
    print("[mo] Hermiticity discriminates: a complex-eps_r (lossy) gyrotropic tensor is non-Hermitian "
          "(viol={:.2f})".format(float(np.max(np.abs(Tlossy - Tlossy.conj().T)))), flush=True)

    # GATE D: end-to-end through the off-diagonal UPML FEM
    gate_d, msg_d = _gate_d_fem()
    print(msg_d, flush=True)

    overall = gate_a and gate_b and gate_c and gate_d
    print("[mo]", flush=True)
    print("[mo] GATE A (tensor eigenvalues eps_r, eps_r +/- g): {}".format(
        "PASS" if gate_a else "FAIL"), flush=True)
    print("[mo] GATE B (circular-TMM rotation == analytic theta_F within {:.0%} rel; g=0 -> 0): "
          "{}".format(ROT_RTOL, "PASS" if gate_b else "FAIL"), flush=True)
    print("[mo] GATE C (gyrotropic tensor is HERMITIAN = independent losslessness, catches off-diag "
          "sign; a complex-eps_r variant fails it): {}".format("PASS" if gate_c else "FAIL"), flush=True)
    print("[mo] GATE D (off-diagonal UPML FEM == circular-eigenmode Jones-TMM, lossless): {}".format(
        "PASS" if gate_d else "FAIL"), flush=True)
    print("[mo] *** MAGNETO-OPTIC FARADAY (gyrotropic tensor): {} ***".format(
        "PASS" if overall else "FAIL"), flush=True)
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
