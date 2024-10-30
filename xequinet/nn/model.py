from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .basic import compute_edge_data, compute_properties
from .ewald import EwaldBlock, EwaldInitialNonPBC, EwaldInitialPBC
from .output import resolve_output
from .xpainn import XEmbedding, XPainnMessage, XPainnUpdate


class BaseModel(nn.Module):

    cutoff_radius: float
    module_list: nn.ModuleList
    extra_properties: List[str]

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_forces: bool = True,
        compute_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        data = compute_edge_data(
            data=data,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
        )
        for module in self.module_list:
            data = module(data)
        result = compute_properties(
            data=data,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
            training=self.training,
            extra_properties=self.extra_properties,
        )
        return result


class XPaiNN(BaseModel):
    """
    eXtended PaiNN.
    """

    def __init__(
        self,
        node_dim: int = 128,
        node_irreps: str = "128x0e + 64x1o + 32x2e",
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux56",
        num_basis: int = 20,
        rbf_kernel: str = "bessel",
        cutoff: float = 5.0,
        cutoff_fn: str = "cosine",
        action_blocks: int = 3,
        activation: str = "silu",
        norm_type: str = "layer",
        output_modes: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.cutoff_radius = cutoff
        self.module_list = nn.ModuleList()
        embed = XEmbedding(
            node_dim=node_dim,
            node_irreps=node_irreps,
            embed_basis=embed_basis,
            aux_basis=aux_basis,
            num_basis=num_basis,
            rbf_kernel=rbf_kernel,
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
        )
        self.module_list.append(embed)
        for _ in range(action_blocks):
            message = XPainnMessage(
                node_dim=node_dim,
                node_irreps=node_irreps,
                num_basis=num_basis,
                activation=activation,
                norm_type=norm_type,
            )
            update = XPainnUpdate(
                node_dim=node_dim,
                node_irreps=node_irreps,
                activation=activation,
                norm_type=norm_type,
            )
            self.module_list.extend([message, update])

        self.extra_properties = []
        if output_modes is None:
            output_modes = ["energy"]
        elif not isinstance(output_modes, Iterable):
            output_modes = [output_modes]
        for mode in output_modes:
            output = resolve_output(mode, **kwargs)
            self.module_list.append(output)
            self.extra_properties.extend(output.extra_properties)


class XPaiNNEwald(XPaiNN):
    """
    XPaiNN with Ewald message passing.
    """

    def __init__(
        self,
        # original XPaiNN parameters
        node_dim: int = 128,
        node_irreps: str = "128x0e + 64x1o + 32x2e",
        embed_basis: str = "gfn2-xtb",
        aux_basis: str = "aux56",
        num_basis: int = 20,
        rbf_kernel: str = "bessel",
        cutoff: float = 5.0,
        cutoff_fn: str = "cosine",
        action_blocks: int = 3,
        activation: str = "silu",
        norm_type: str = "layer",
        output_modes: Optional[Union[str, List[str]]] = None,
        # Ewald parameters
        use_pbc: bool = True,
        num_k_points: Optional[Tuple[int, int, int]] = None,  # pbc
        k_cutoff: Optional[float] = None,  # non-pbc
        delta_k: Optional[float] = None,  # non-pbc
        num_k_basis: Optional[int] = None,  # non-pbc
        k_offset: Optional[float] = None,  # non-pbc
        projection_dim: int = 8,
        ewald_blocks: int = 3,
        ewald_output_modes: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            node_dim=node_dim,
            node_irreps=node_irreps,
            embed_basis=embed_basis,
            aux_basis=aux_basis,
            num_basis=num_basis,
            rbf_kernel=rbf_kernel,
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
            action_blocks=action_blocks,
            activation=activation,
            norm_type=norm_type,
            output_modes=output_modes,
            **kwargs,
        )
        if use_pbc:
            ewald_initial = EwaldInitialPBC(
                num_k_points=num_k_points,
                projection_dim=projection_dim,
            )
        else:
            ewald_initial = EwaldInitialNonPBC(
                k_cutoff=k_cutoff,
                delta_k=delta_k,
                num_k_basis=num_k_basis,
                k_offset=k_offset,
                projection_dim=projection_dim,
            )
        self.module_list.append(ewald_initial)
        for _ in range(ewald_blocks):
            ewald_block = EwaldBlock(
                node_dim=node_dim,
                projection_dim=projection_dim,
                activation=activation,
                norm_type=norm_type,
            )
            self.module_list.append(ewald_block)
        if ewald_output_modes is None:
            ewald_output_modes = ["energy"]
        elif not isinstance(ewald_output_modes, Iterable):
            ewald_output_modes = [ewald_output_modes]
        for mode in ewald_output_modes:
            output = resolve_output(mode, **kwargs)
            self.module_list.append(output)
            self.extra_properties.extend(output.extra_properties)


def resolve_model(model_name: str, **kwargs) -> BaseModel:
    models_factory = {
        "xpainn": XPaiNN,
        "xpainn-ewald": XPaiNNEwald,
    }
    if model_name.lower() not in models_factory:
        raise NotImplementedError(f"Unsupported model {model_name}")
    return models_factory[model_name.lower()](**kwargs)
