# qr-plugin-ase
A plugin for the Atomic Simulation Environment(ASE) (https://wiki.fysik.dtu.dk/ase/)

Quantum refinement code Q|R (https://github.com/qrefine/qrefine/blob/master/restraints.py) calls QM calculator through ASE.

DFTB+ notes:
The DFTB+ plugin (`dftbplus_qr.py`) takes `quantum.method` as `gfn1|gfn2`
combined with `alpb|gbsa|none`, for example `gfn2/alpb` (the default) or
`gfn1/none`. ALPB and GBSA use water and explicitly select tblite-compatible
COSMO radii and the P16 kernel. No external parameter files are required.
Classic SCC-DFTB with Slater-Koster (`.skf`) parameter sets (e.g. `3ob`,
`mio` from dftb.org) is not wired up: conda-forge's `dftbplus` package does
not ship any `.skf` files. User need to write a custom `hamiltonian_params` dict
to use a non-xtb hamiltonian. mpi wrapper is vendor agnostic.
