from typing import Dict, Final, List, Set

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
# MULTI_GRAPH_KEYS: Final[List[str]] = [BATCH, BATCH_PTR, NUM_GRAPHS]
# keys for long-range interactions
LONG_EDGE_INDEX: Final[str] = "long_edge_index"
LONG_EDGE_LENGTH: Final[str] = "long_edge_length"

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

GRAD_PROPERTIES: Final[Set[str]] = {  # properties that are gradients got by autograd
    FORCES,
    BASE_FORCES,
    VIRIAL,
}
BASE_PROPERTIES: Final[Dict[str, str]] = {  # properties that are base properties
    BASE_ENERGY: TOTAL_ENERGY,
    BASE_FORCES: FORCES,
    BASE_CHARGES: ATOMIC_CHARGES,
    BASE_DIPOLE: DIPOLE,
}
STANDARD_PROPERTIES: Final[
    Set[str]
] = {  # properties that can be printed when inference
    TOTAL_ENERGY,
    FORCES,
    VIRIAL,
    DIPOLE,
    POLARIZABILITY,
}


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
