# Installing HYPRE for NGSolve on Windows

This guide tells a Windows user how to obtain an NGSolve build whose
`ngsolve.Preconditioner(bf, "hypre")` (BoomerAMG) or the HYPRE-AMS
auxiliary-space Maxwell preconditioner actually works, so that DynaMeta's
`linear_solver="hypre"` / `linear_solver="ams"` option becomes usable instead
of silently falling back to BDDC.

ASCII-only, plain hyphens, cp1252-safe.

---

## Read this first: honest support status

Native-Windows HYPRE + NGSolve is NOT officially supported in practice. There
is no native-Windows pip wheel, no native-Windows MPI/HYPRE build
documentation, and no native-Windows HYPRE tutorial from the NGSolve project.

The facts that drive this guide:

- The standard `pip install ngsolve` wheel (including `win_amd64`) is a SERIAL,
  non-MPI build. It contains no MPI and therefore no working HYPRE interface.
- In NGSolve, HYPRE is fundamentally an MPI-parallel interface. HYPRE support
  is gated behind `-DUSE_HYPRE=ON` (default OFF) AND an MPI-enabled build
  (`-DUSE_MPI=ON`); HYPRE's MPI define (`HYPRE_WITH_MPI`) is added by NGSolve
  only when `NETGEN_USE_MPI` is set.
- Every piece of official NGSolve documentation, tutorials, and packaging
  treats parallel NGSolve as a Linux/macOS feature. The only Windows-targeted
  parallel build guidance from the project is a forum post on building under
  WSL2 + Ubuntu.
- The native NGSolve-HYPRE interface (`type="hypre"`) only works with MPI and
  requires at least 2 ranks. Running at `-np 1` throws `std::bad_cast`. It also
  "only really works well for order 1 discretizations" (a generic AMG property,
  per the NGSolve developers on the forum).
- DynaMeta's own behavior reflects all of this: by default it FALLS BACK to
  BDDC because the standard wheel lacks HYPRE, and naively requesting an
  unavailable preconditioner SEGFAULTS, so the real attempt is gated behind the
  env var `DYNAMETA_AMG_OK=1`.

Because of this, the recommended route is WSL2 (Linux) FIRST. The native
Windows path is documented afterward for those who specifically need it, with
explicit caveats about which steps are unverified.

---

## Decision guide: pick your route

| Route | Reliability | Runs serial (1 rank)? | Gets AMS? | Notes |
|---|---|---|---|---|
| WSL2 + PETSc-HYPRE via ngsPETSc | Most reliable; project-tested | Yes | Yes (boomeramg/ams/ads) | Recommended for most users |
| WSL2 + native ngsHypre | Reliable | No (needs >= 2 ranks) | No (BoomerAMG only) | Lighter, but MPI-only |
| Native Windows source build with MPI+HYPRE | Fragile / unsupported | Depends on interface | In-tree build only | Only if you cannot use WSL |

---

## Route A (RECOMMENDED): WSL2 + Linux NGSolve with HYPRE

This is the path the NGSolve project itself documents for parallel work on
Windows. You run a Linux NGSolve build inside WSL2; the Windows pip wheel is
not involved.

### A.1 Prerequisites

1. Windows 10 or 11 with WSL2 enabled. In an elevated PowerShell:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
   Reboot if prompted, then launch the Ubuntu shell and create your user.

2. Inside Ubuntu, the usual build toolchain (compiler, CMake, Git, Python 3,
   an MPI implementation). The community WSL build instructions target Ubuntu
   22.04.

### A.2 Two sub-options once you are inside WSL

You build an MPI-enabled NGSolve, then get HYPRE on top of it one of two ways.

#### A.2a PETSc-HYPRE via ngsPETSc (recommended sub-option)

PETSc's `PCHYPRE` exposes BoomerAMG, AMS, and ADS, and it runs in serial OR
parallel (one rank works), which is far friendlier for development and
order-by-order debugging.

- Build NGSolve from source with `-DUSE_MPI=ON`.
- Configure PETSc with `--download-hypre`.
- Install ngsPETSc (per its install docs).
- In Python, request HYPRE through PETSc:
  ```python
  pre = Preconditioner(a, "PETScPC", pc_type="hypre")
  ```

Hard requirement: NGSolve, PETSc, petsc4py, and mpi4py must all be built
against ONE consistent MPI. Do not mix the Windows pip wheel with a WSL PETSc.

#### A.2b Native ngsHypre add-on (lighter, BoomerAMG only)

On top of an MPI-enabled NGSolve:
```bash
pip3 install git+https://github.com/NGSolve/ngsHypre.git
mpirun -np 4 python3 -m ngs_hypre.demos.example1
```

Limitations (stated by the maintainers): works ONLY with MPI, needs at least 2
ranks (`-np 1` -> `std::bad_cast`), and works well only for order-1
discretizations. AMS is not exposed through ngsHypre.

Note: if instead you build NGSolve itself in-tree with `-DUSE_HYPRE=ON
-DUSE_MPI=ON`, both the `"hypre"` (BoomerAMG) and `"hypre_ams"` (HCurl AMS)
factory strings become available without the separate ngsHypre package. The
standalone ngsHypre pip package ships only `"hypre"`.

---

## Route B (NATIVE WINDOWS, unsupported/fragile): build NGSolve with MPI + HYPRE

WARNING: This path is not officially supported and is fragile. Nothing in the
NGSolve CMake hard-blocks Windows MPI, but there is no official native-Windows
recipe, and several steps below are version-dependent or only community-tested.
Steps flagged UNVERIFIED are not confirmed end-to-end by primary sources for a
native MSVC NGSolve build. Prefer Route A unless you cannot use WSL.

### B.1 Prerequisites (install in this order)

1. Visual Studio 2022 or VS Build Tools with the MSVC C/C++ toolset. NGSolve
   docs advise Visual Studio 2022. HYPRE is C, so the C++ workload's C compiler
   suffices.
2. CMake 3.21 or newer (HYPRE's hard minimum is 3.21; check "Add CMake to the
   system PATH for all users" in the installer).
3. Git.
4. Python 3, x86-64 installer, with "Add Python 3 to PATH". The standard
   Windows Python installer includes the include/libs needed for the embedded
   build.
5. Microsoft MPI (MS-MPI) -- BOTH the runtime and the SDK (see B.2).

### B.2 Install Microsoft MPI (runtime + SDK)

MS-MPI is the standard Windows MPI provider and the one CMake's `FindMPI`
auto-detects. You need BOTH pieces; the SDK is what `find_package(MPI)`
locates.

Latest verified release: MS-MPI v10.1.3, build 10.1.12498.52 (released
2024-07-15). Use v10.x; older MS-MPI predates MPI-3 and lacks symbols NGSolve
expects.

Download both files from the Microsoft Download Center (id=105289):
https://www.microsoft.com/en-us/download/details.aspx?id=105289
(landing page: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi )

- `msmpisetup.exe` (~7.4 MB): runtime -- `mpiexec.exe`, `smpd.exe`, `msmpi.dll`.
- `msmpisdk.msi` (~2.2 MB): SDK -- `mpi.h`, `msmpi.lib`, `mpif.h`.

Install steps:

1. Run `msmpisetup.exe` first (runtime), accept defaults.
2. Run `msmpisdk.msi` (SDK), accept defaults.
3. Open a NEW shell so the installer-set environment variables are visible.
   Existing shells / IDEs will not see them until restarted.
4. Verify in PowerShell:
   ```powershell
   mpiexec -help
   $env:MSMPI_INC
   $env:MSMPI_LIB64
   ```

Default install locations:

- Runtime: `C:\Program Files\Microsoft MPI\` (binaries in `...\Bin\`)
- SDK: `C:\Program Files (x86)\Microsoft SDKs\MPI\` (headers in `\Include\`,
  libs in `\Lib\x64\` and `\Lib\x86\`)

Environment variables set by the SDK installer (exact default values):
```
MSMPI_BIN   = C:\Program Files\Microsoft MPI\Bin\
MSMPI_INC   = C:\Program Files (x86)\Microsoft SDKs\MPI\Include\
MSMPI_LIB32 = C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x86\
MSMPI_LIB64 = C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\
```
`MSMPI_INC` and `MSMPI_LIB64` are the only mechanism CMake uses to locate the
SDK. Build x64 so the bitness matches `Lib\x64`.

Silent/CI install:
```
msmpisetup.exe -unattend -full
msiexec /i msmpisdk.msi /quiet /qn
```

### B.3 (Optional) Build HYPRE standalone first

NGSolve's SuperBuild can auto-download and build HYPRE when you pass
`-DUSE_HYPRE=ON` without a prebuilt `HYPRE_DIR`. If you instead want to build
HYPRE yourself (for example to control flags or to supply `HYPRE_DIR`),
HYPRE's CMake build is its officially supported Windows path and works with
MSVC.

Verified facts about HYPRE's CMake build:

- CMake 3.21+ is required; out-of-source builds are mandatory.
- The `CMakeLists.txt` lives in the `src/` subdirectory, so use `-S src`.
- The current option name is `HYPRE_ENABLE_MPI` (default ON). Do NOT use the
  legacy `HYPRE_WITH_MPI` spelling on a current checkout -- it is the old
  autotools `--with-MPI` convention and is silently ignored by current CMake
  (CMake does not error on unknown `-D` cache vars). NGSolve's own build adds
  the C macro `HYPRE_WITH_MPI`; that is a compile define, not the CMake option.
- `BUILD_SHARED_LIBS` is OFF by default (static). `HYPRE_ENABLE_OPENMP` OFF,
  `HYPRE_ENABLE_FORTRAN` ON (only builds Fortran examples/tests, which are off
  by default, so a pure-MSVC C build is fine; set
  `-DHYPRE_ENABLE_FORTRAN=OFF` if Fortran probing causes trouble).

Serial HYPRE (simplest, most reliable):
```bat
git clone https://github.com/hypre-space/hypre.git
cd hypre
cmake -S src -B build ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DHYPRE_ENABLE_MPI=OFF ^
  -DCMAKE_INSTALL_PREFIX=C:\libs\hypre
cmake --build build --config Release --target INSTALL
```
NOTE: a serial HYPRE will NOT satisfy NGSolve's HYPRE interface, which requires
MPI. Build with MPI (below) if you intend to wire it into NGSolve.

Parallel HYPRE with MS-MPI:
```bat
cmake -S src -B build ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DHYPRE_ENABLE_MPI=ON ^
  -DCMAKE_INSTALL_PREFIX=C:\libs\hypre
cmake --build build --config Release --target INSTALL
```
If `FindMPI` does not auto-detect MS-MPI, point it at the include dir and
`msmpi.lib` (NOT at `cl.exe` -- MS-MPI has no compiler wrapper):
```bat
  -DMPI_C_INCLUDE_PATH="C:/Program Files (x86)/Microsoft SDKs/MPI/Include" ^
  -DMPI_C_LIBRARIES="C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib"
```
Alternatively force the guesser: `-DMPI_GUESS_LIBRARY_NAME=MSMPI`.

Other ways to get HYPRE on Windows (both community-maintained, not by the HYPRE
team):

- vcpkg: `vcpkg install hypre` -- version 2.32.0, but STATIC-ONLY on Windows
  (no DLL), MSVC ABI, and MPI resolves to msmpi which must be installed
  system-wide. The `x64-windows-static` triplet has a documented history of
  failing to find MSMPI.
- MSYS2/MinGW: `pacman -S mingw-w64-x86_64-hypre` -- version 2.33.0, GCC/MinGW
  ABI (do NOT link into an MSVC program). This will not match an MSVC NGSolve.

### B.4 Build NGSolve from source with MPI + HYPRE

Follow the official Windows source-build layout (base dir with `src`, `build`,
`install`). The official Windows page documents NO MPI or HYPRE flags -- the
flags below come from the NGSolve CMake source and the HYPRE-support forum
thread, and are the same flags used on Linux:

```bat
cmake ..\src ^
  -DUSE_MPI=ON ^
  -DUSE_HYPRE=ON ^
  -DCMAKE_INSTALL_PREFIX="BASEDIR\install"
cmake --build . --config Release --target install
cmake --build . --config Release --target set_environment_variables
```

- `-DUSE_MPI=ON` is propagated to the Netgen subproject by NGSolve's
  SuperBuild; HYPRE's MPI define is gated on `NETGEN_USE_MPI`.
- `-DUSE_HYPRE=ON` enables the HYPRE interface and, without a prebuilt
  `HYPRE_DIR`, pulls in `cmake/external_projects/hypre.cmake` to auto-build
  HYPRE. To use a HYPRE you built in B.3, also pass `-DHYPRE_DIR=C:\libs\hypre`.
- The `set_environment_variables` target sets PATH, NETGENDIR, PYTHONPATH.

UNVERIFIED: This exact native-Windows MSVC invocation of `-DUSE_MPI=ON
-DUSE_HYPRE=ON` is not documented or endorsed by the NGSolve project for
Windows; it is the documented Linux/WSL recipe applied on Windows. Expect
friction in making NGSolve, Netgen, and the HYPRE subproject all agree on
MS-MPI under MSVC. If it does not configure cleanly, fall back to Route A.

---

## Verify it worked

The goal of verification is to confirm that constructing the HYPRE
preconditioner does NOT segfault on a tiny mesh. Because the native
NGSolve-HYPRE interface is MPI-only and needs at least 2 ranks, run the snippet
under `mpiexec` / `mpirun` with 2 ranks.

Save as `verify_hypre.py`:

```python
from mpi4py import MPI
from ngsolve import *

comm = MPI.COMM_WORLD

# Tiny distributed mesh.
ngmesh = unit_square.GenerateMesh(maxh=0.3, comm=comm)
mesh = Mesh(ngmesh)

# BoomerAMG applies to scalar H1, order 1.
# HYPRE requires a fully-stored (non-symmetric) matrix, hence symmetric=False.
fes = H1(mesh, order=1, dirichlet=".*")
u, v = fes.TnT()
a = BilinearForm(grad(u) * grad(v) * dx, symmetric=False)

# If this line does not segfault, the HYPRE preconditioner is available.
pre = Preconditioner(a, "hypre")

a.Assemble()
pre.Update()   # triggers HYPRE_BoomerAMGSetup

if comm.rank == 0:
    print("OK: ngsolve.Preconditioner(bf, 'hypre') built without segfault")
```

Run it (native Windows MS-MPI, or inside WSL):
```
mpiexec -n 2 python verify_hypre.py
```
(On Linux/WSL use `mpirun -np 2 python3 verify_hypre.py`.)

Expected: it prints the OK line and exits cleanly. A `std::bad_cast` means you
ran at a single rank -- use at least 2. A segfault or "preconditioner type not
registered" means the NGSolve build does not actually contain HYPRE; revisit
the build (Route A or B), do not set `DYNAMETA_AMG_OK=1`.

To verify the AMS (HCurl) path on an in-tree `USE_HYPRE` NGSolve build, replace
the space and form with HCurl and request `"hypre_ams"`:
```python
fes = HCurl(mesh, order=0, dirichlet=".*")   # must be HCurlHighOrderFESpace
u, v = fes.TnT()
a = BilinearForm(curl(u) * curl(v) * dx + u * v * dx, symmetric=False)
pre = Preconditioner(a, "hypre_ams")          # AMS auto-builds gradient + coords
```
AMS enforces an HCurl space at construction (otherwise it throws
"HYPRE-AMS Setup needs HCurlHighOrder-FESpace"). The `"hypre_ams"` string is
available only from an in-tree `USE_HYPRE` build, NOT from the pip ngsHypre
add-on. For PETSc-HYPRE (Route A.2a), use
`Preconditioner(a, "PETScPC", pc_type="ams")` instead.

---

## Caveats / known pitfalls

- MPI-only, 2+ ranks: the native `"hypre"` / `"hypre_ams"` interfaces assume
  ParallelMatrix/ParallelDofs and DO NOT run serially. `-np 1` throws
  `std::bad_cast`. Always launch with `mpiexec/mpirun -n 2` or more.
- Fully-stored matrix required: build the BilinearForm with `symmetric=False`,
  or the HYPRE classes throw "Please use fully stored sparse matrix for hypre".
- Order-1 only (practically): BoomerAMG "only really works well for order 1
  discretizations" per the NGSolve developers (MEDIUM confidence -- single
  forum source, but consistent with standard AMG behavior). For higher order,
  wrap with BDDC or an auxiliary-space/p-multigrid outer scheme.
- The standard pip wheel never has HYPRE: do not expect `pip install ngsolve`
  on Windows (or even the default native VS source build) to provide HYPRE/MPI.
  Both are serial.
- AMS is NOT in the pip ngsHypre add-on: ngsHypre ships only BoomerAMG
  (`"hypre"`). AMS is available only via an in-tree `USE_HYPRE` NGSolve build,
  or via PETSc PCHYPRE (`pc_type="ams"`).
- Fresh shell after MS-MPI install: IDEs/CMake launched before installing
  MS-MPI will not see `MSMPI_*` env vars. Restart them.
- Bitness mismatch is the #1 "found MPI but link fails" cause: build x64 so it
  matches `Lib\x64` / `MSMPI_LIB64`.
- HYPRE flag name: use `-DHYPRE_ENABLE_MPI=ON`, NOT `-DHYPRE_WITH_MPI`. The
  latter is legacy autotools naming and is silently ignored by current CMake.
- HYPRE shared-lib history: current releases honor `-DBUILD_SHARED_LIBS=ON`
  alone, but older tags (around v2.20) had a missing Windows DLL RUNTIME
  install destination and sometimes required both `BUILD_SHARED_LIBS=ON` and
  the legacy `HYPRE_ENABLED_SHARED=ON`. For static linking (the easy default),
  leave `BUILD_SHARED_LIBS` OFF.
- MS-MPI deprecation errors under MSVC (UNVERIFIED workaround): users report
  defining `MSMPI_NO_DEPRECATE_20` (e.g. add
  `-DCMAKE_C_FLAGS="/DMSMPI_NO_DEPRECATE_20"`). This is user-reported, not in
  official HYPRE docs.
- FindMPI hint-variable spellings are version-dependent (MEDIUM confidence):
  `MPI_C_INCLUDE_PATH` / `MPI_C_LIBRARIES` are older but widely-working hints;
  newer CMake prefers the `MPI_HOME` env var plus auto-detect.
- vcpkg HYPRE is static-only on Windows (no DLL), version 2.32.0, and its
  static triplet has a documented history of failing to find MSMPI.
- MSYS2/MinGW HYPRE is GCC ABI -- do not link it into an MSVC NGSolve.
- UNVERIFIED end to end: the entire native-Windows Route B (especially the
  NGSolve `-DUSE_MPI=ON -DUSE_HYPRE=ON` MSVC configure in B.4) is not confirmed
  by any primary source for native Windows; it is the Linux/WSL recipe applied
  on Windows. If you hit a wall, use Route A (WSL2).

---

## Enabling HYPRE / AMS in DynaMeta

Once you have a HYPRE-enabled NGSolve in place and the verify snippet above
runs without segfaulting, enable the DynaMeta path by setting the gate env var:

PowerShell:
```powershell
$env:DYNAMETA_AMG_OK = "1"
```
cmd:
```bat
set DYNAMETA_AMG_OK=1
```
WSL / bash:
```bash
export DYNAMETA_AMG_OK=1
```

With `DYNAMETA_AMG_OK=1` set, DynaMeta will actually attempt the requested
preconditioner instead of falling back to BDDC, so `linear_solver="hypre"`
(BoomerAMG) and `linear_solver="ams"` (HCurl auxiliary-space Maxwell) become
usable. Leave the gate UNSET on any environment where the NGSolve build lacks
HYPRE -- requesting an unavailable preconditioner segfaults, which is exactly
what the gate plus the BDDC fallback exist to prevent.
