from typing import Final, Literal

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

# Results
ATOMIC_ENERGIES: Final[str] = "atomic_energies"
TOTAL_ENERGY: Final[str] = "energy"
FORCES: Final[str] = "forces"
VIRIAL: Final[str] = "virial"
STRESS: Final[str] = "stress"
ATOMIC_CHARGES: Final[str] = "atomic_charges"
TOTAL_CHARGE: Final[str] = "charge"
