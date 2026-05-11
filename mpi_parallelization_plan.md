# MPI Parallelization Plan for `topology_optimization.py`

## Overview of Bottlenecks

The main costs per iteration are:

1. Stiffness matrix **assembly** (`Kes → K`)
2. **Linear system solve** (`K u = f`)
3. **Sensitivity calculation** (`dobj` per element)
4. **Filter** application (`H @ x`)

Items 1, 3, and 4 are embarrassingly parallel over elements. Item 2 requires a
distributed sparse solver.

---

## Phase 1 — MPI Setup & Element Partitioning

- Initialize `MPI.COMM_WORLD`, obtain `rank` and `size`.
- Compute a local element range per rank:
  `el_start, el_end = np.array_split(np.arange(n), size)[rank][[0,-1]]`
- Keep the full `edofMat` on all ranks (it is small), but each rank processes
  only its owned element slice.
- Gate all I/O (logging, VTK export, matplotlib) behind `if rank == 0`.

---

## Phase 2 — Parallel Stiffness Assembly

- Each rank computes `Kes_local` for its owned elements and the corresponding
  COO triplets `sK_local`, `iK_local`, `jK_local`.
- Use `comm.Allgatherv` (variable-length gather) to collect COO triplets on all
  ranks before assembling `K`, **or** use `PETSc.Mat` distributed assembly
  directly (preferred — see Phase 3).

---

## Phase 3 — Distributed Linear Solve (Core Change)

- Replace `scipy.sparse.linalg.factorized` / `solve_lin` with `petsc4py` +
  MUMPS or a parallel iterative solver (e.g., PETSc GMRES/CG + ILU
  preconditioner).
- Create a `PETSc.KSP` object **once** outside the loop and reuse the same
  symbolic factorization pattern across iterations (only a numeric refactor is
  needed each iteration).
- Wrap the PETSc solve in the existing `solve_lin` interface so the rest of the
  code is unaffected.

---

## Phase 4 — Parallel Sensitivity Calculation

- Each rank computes `dobj_local` for its owned element slice using local `adj`,
  `u`, and `Kes`.
- `comm.Allreduce(dobj_local, dobj, op=MPI.SUM)` assembles global sensitivities
  on all ranks.
- Apply the same pattern to density body-force contributions.

---

## Phase 5 — Parallel Matrix Filter

- Distribute rows of `H` across ranks so each rank owns `H_local`.
- Perform the local matrix-vector product, then use `comm.Allgatherv` to
  reassemble the full filtered `xPhys`.
- For the Helmholtz filter: the filter PDE solve can also be routed through
  PETSc.

---

## Phase 6 — Optimizer (Rank 0 or Replicated)

- MMA / OC are computationally cheap and operate on design variables only.
- Run on rank 0, then `comm.Bcast(x, root=0)` to distribute the updated `x`
  to all ranks.
- History arrays (`xhist`, etc.) are kept only on rank 0.

---

## Phase 7 — Output & Convergence

- **VTK export**: only rank 0 writes; gather `xPhys` and `u` to rank 0 first.
- **Convergence check**: compute the local change metric, then
  `comm.Allreduce` with `MPI.MAX` to obtain the global value.

---

## Suggested Implementation Order

| Step | Phase | Rationale |
|------|-------|-----------|
| 1 | Phase 1 | Trivial; establishes the skeleton |
| 2 | Phase 3 | Largest single speedup; requires `petsc4py` |
| 3 | Phase 4 | Parallel sensitivities; depends on Phase 1 |
| 4 | Phase 2 | Parallel assembly feeding Phase 3 |
| 5 | Phase 5 | Parallel filter |
| 6 | Phase 6 + 7 | I/O cleanup and convergence broadcast |

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `mpi4py` | MPI bindings (point-to-point and collective operations) |
| `petsc4py` | Distributed sparse matrices and solvers (wraps PETSc) |
| `petsc4py` (MUMPS) | Parallel sparse direct solver via PETSc interface |

---

## Main Risk

The DOF numbering produced by `create_edofMat` uses a global numbering scheme.
PETSc requires this to be mapped to a distributed index set. Budget extra time
for the **global-to-local DOF mapping** and ghost DOF bookkeeping at process
boundaries.
