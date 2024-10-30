from typing import Dict, Final, Set

# basic keys in datapoints
POSITIONS: Final[str] = "pos"
ATOMIC_NUMBERS: Final[str] = "atomic_numbers"
EDGE_INDEX: Final[str] = "edge_index"
CELL_OFFSETS: Final[str] = "cell_offsets"
CELL: Final[str] = "cell"
PBC: Final[str] = "pbc"
# keys for collated batches
BATCH: Final[str] = "batch"
BATCH_PTR: Final[str] = "ptr"
NUM_GRAPHS: Final[str] = "num_graphs"

# intermediate variable
CENTER_IDX: Final[int] = 0
NEIGHBOR_IDX: Final[int] = 1
EDGE_LENGTH: Final[str] = "edge_length"
EDGE_VECTOR: Final[str] = "edge_vector"
STRAIN: Final[str] = "strain"

RADIAL_BASIS_FUNCTION: Final[str] = "radial_basis_function"
ENVELOPE_FUNCTION: Final[str] = "envelope_function"
SPHERICAL_HARMONICS: Final[str] = "spherical_harmonics"
NODE_INVARIANT: Final[str] = "node_invariant"
NODE_EQUIVARIANT: Final[str] = "node_equivariant"

# Ewald message passing
K_DOT_R: Final[str] = "k_dot_r"
SINC_DAMPING: Final[str] = "sinc_damping"
DOWN_PROJECTION: Final[str] = "down_projection"

# Properties
ATOMIC_ENERGIES: Final[str] = "atomic_energies"
TOTAL_ENERGY: Final[str] = "energy"
BASE_ENERGY: Final[str] = "base_energy"
ENERGY_PER_ATOM: Final[str] = "energy_per_atom"
FORCES: Final[str] = "forces"
BASE_FORCES: Final[str] = "base_forces"
VIRIAL: Final[str] = "virial"
STRESS: Final[str] = "stress"
ATOMIC_CHARGES: Final[str] = "atomic_charges"
BASE_CHARGES: Final[str] = "base_charges"
TOTAL_CHARGE: Final[str] = "charge"

DIPOLE: Final[str] = "dipole"
BASE_DIPOLE: Final[str] = "base_dipole"
DIPOLE_MAGNITUDE: Final[str] = "dipole_magnitude"
POLARIZABILITY: Final[str] = "polarizability"
ISO_POLARIZABILITY: Final[str] = "iso_polarizability"

# properties that are gradients got by autograd
GRAD_PROPERTIES: Final[Set[str]] = {
    FORCES,
    BASE_FORCES,
    VIRIAL,
}
# properties that are base properties
BASE_PROPERTIES: Final[Dict[str, str]] = {
    BASE_ENERGY: TOTAL_ENERGY,
    BASE_FORCES: FORCES,
    BASE_CHARGES: ATOMIC_CHARGES,
    BASE_DIPOLE: DIPOLE,
}
# properties that can be printed when inference
STANDARD_PROPERTIES: Final[Set[str]] = {
    TOTAL_ENERGY,
    FORCES,
    VIRIAL,
    DIPOLE,
    POLARIZABILITY,
}
# vector properties
VECTOR_PROPERTIES: Final[Set[str]] = {DIPOLE}
# atomic vector properties
ATOMIC_VECTOR_PROPERTIES: Final[Set[str]] = {FORCES}

SPATIAL_EXTENT: Final[str] = "spatial_extent"

# general-purpose properties
SCALAR_OUTPUT: Final[str] = "scalar_output"
CARTESIAN_TENSOR: Final[str] = "cartesian_tensor"

# xTB methods
xTB_METHODS: Final[Dict[str, str]] = {
    "gfn1-xtb": "GFN1-xTB",
    "gfn2-xtb": "GFN2-xTB",
}

# others
TRAIN: Final[str] = "train"
VALID: Final[str] = "valid"
TEST: Final[str] = "test"

# for lammps interface
CUTOFF_RADIUS: Final[str] = "cutoff_radius"
JIT_FUSION_STRATEGY: Final[str] = "jit_fusion_strategy"
N_SPECIES: Final[str] = "n_species"
PERIODIC_TABLE: Final[str] = "periodic_table"
