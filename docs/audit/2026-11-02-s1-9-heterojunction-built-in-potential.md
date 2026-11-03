# S1-9: heterojunction built-in potential (equilibrium fixed; DD/bipolar design note)

Audit row: `docs/audit/2026-07-17-exhaustive-audit.md` S1-9 (P3, physics). Wave 7, 2026-11-02.

## What was wrong (equilibrium)

The 3-D equilibrium closure is `n = N_c F_1/2((psi - Phi_c)/V_t)` with the Fermi level at 0, so the
conduction edge is `E_c = -q psi + q Phi_c`. `physics_equilibrium.setup_phi_c0` calibrates `Phi_c`
PER REGION so that region is neutral at `psi = 0` (`Phi_c0(r) = -eta_bg(r) V_t`). With
Potential-continuity interfaces that is an implicit electron affinity `chi_r = eta_bg(r) kT` for every
region -- all work functions equal -- so `psi = 0` was the exact equilibrium of ANY stack:

* no built-in potential at a heterojunction (no band offset from the affinities), and
* none at a same-material n+/n HOMOjunction either -- the per-region calibration silently inserted a
  conduction-band STEP of `kT (eta_bg,1 - eta_bg,2)` that exactly cancels the physical built-in
  potential `V_t ln(n1/n2)` (Boltzmann limit; FD in general).

## The fix (equilibrium, self-contained)

Equilibrium needs only Fermi-level alignment, which Potential continuity already enforces once the
band edges are referenced to a common vacuum level. `Stacked3DSpec.band_alignment()` applies the
Anderson rule:

    Phi_c(r)       = Phi_c0(semi) + (chi_semi - chi_r) / q
    psi_neutral(r) = Phi_c(r) - Phi_c0(r) = (W_semi - W_r) / q,   W = chi + (E_c - E_F)_neutral

and `Devsim3DEquilibrium.build_device` sets each extra region's `Phi_c0` parameter to `Phi_c(r)` (the
primary's is unchanged, so single-semiconductor solves are byte-identical) and seeds its Potential at
`psi_neutral(r)`. Inputs: `Stacked3DSpec.chi_semi_eV` and an optional 6th/7th field per
`extra_semiconductors` entry, `(..., n_bg_m3, chi_eV, dos_mass_kg)`; `from_design` fills them from each
layer's `TransportModel.chi_eV` (now load-bearing for heterostacks) and DOS mass. Same material ->
zero offset by definition; a DIFFERENT material without both affinities raises at construction (there
is no neutral default for a band offset). The gate electrode keeps the BODY semiconductor's work
function (flat band of the body at `Vg = 0`), as before.

Gates: `tests/test_audit_2026_07_17_wave7_carriers.py::test_s1_9_*` (the alignment algebra vs the
closed-form Boltzmann/FD built-in potential; the refusal; build_device installs the aligned `Phi_c`)
and `validation/carriers_3d_multisemi.py` GATE C (3-D solve: a 4e26 / 4e25 m^-3 ITO n+/n stack biased
to the n side's flat band holds both bulks at their neutral potentials).

Sibling not changed here: the 2-D `LayeredDevsimBuilder` meshes adjacent equilibrium semiconductor
layers with the same per-region calibration, so it has the same zero-built-in property. Its contacts
sit on individual semiconductor layers (each pinned at `psi = bias` in ITS OWN neutral frame), so the
2-D fix also has to re-reference every ohmic contact by `psi_neutral(r)` -- the bipolar `phi_bi`
pattern -- and is left to the DEVSIM phase below.

## Design for the DEVSIM drift-diffusion / bipolar phase (deferred)

The DD/bipolar heterostack stays REFUSED (`Stacked3DSpec.__post_init__`), because the interface wires
Potential only and the carrier densities would see natural zero-flux walls (a silently BLOCKING
junction). What that phase needs, in order:

1. **Band edges as node models.** Per region: `Ec = -Potential + Phi_c(r)` and (bipolar)
   `Ev = Ec - Eg(r)/q` from `chi_r` and `Eg(r)` (`TransportModel.band_gap_eV`), with `N_c(r)`, `N_v(r)`
   from that region's DOS masses. Reuse `band_alignment()` for `Phi_c(r)`; it is the equilibrium limit
   of everything below.
2. **Interface carrier equations (thermionic emission, Schroeder/Yang 1993; Horio-Yanai).** For each
   semiconductor-semiconductor interface, replace natural continuity with a flux condition on the
   Electrons (and Holes) interface equation:

       J_n = A*_n T^2 / (q N_c,hi) [ n_lo-side * exp(-dEc / kT) - n_hi-side ]   (sign per edge)

   where `dEc = |Ec(r0) - Ec(r1)|` at the interface node and `A*_n = 4 pi q m*_n k^2 / h^3` uses the
   mass of the side the carriers are emitted FROM. In DEVSIM this is an `interface_model` with explicit
   `@r0/@r1` derivatives, recorded via `eq_registry.record_interface_equation(type="fluxterm")`, and the
   Electrons variable must be DISCONTINUOUS across the interface (it is: each region owns its node
   copy). Potential stays `type="continuous"`.
3. **Degenerate correction.** ITO-class layers are degenerate (`eta ~ 10`): the Boltzmann TE form must
   use the FD quasi-Fermi formulation (`n = N_c F_1/2(eta)`, the flux written in terms of
   `F_1/2^{-1}(n/N_c)`), consistent with the FD-enhanced SG bulk current (`g_enh`). Gate it against
   the equilibrium limit (zero bias -> zero interface flux, density jump = `band_alignment`).
4. **Contacts.** Ohmic contacts pin Potential at `bias + psi_neutral(r)` of the region they sit on
   (generalises `BP.ohmic_built_in_offset_V`); a gate on a dielectric keeps the body reference.
5. **Oracles.** (a) Zero bias reproduces the equilibrium solve of this wave exactly. (b) An n-N
   isotype heterojunction I-V against the analytic thermionic-emission diode
   `J = J_s [exp(qV/kT) - 1]` in the abrupt-junction, low-injection limit. (c) Same-material split
   (zero offset) reduces to the single-region DD current (the GATE A of `carriers_3d_multisemi`, under
   bias). (d) Current conservation `|I_body + I_gate|` at round-off.
