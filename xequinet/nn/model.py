from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn

from .basic import compute_edge_data, compute_properties
from .electronic import ChargeEmbedding, SpinEmbedding
from .ewald import EwaldBlock, EwaldInitialNonPBC, EwaldInitialPBC
from .output import resolve_output
from .xpainn import XEmbedding, XPainnMessage, XPainnUpdate


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mods = nn.ModuleDict()
        self.extra_properties = []

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
        for mod in self.mods.values():
            data = mod(data)
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

    def __init__(self, **kwargs) -> None:
        super().__init__()
        # solve hyperparameters
        node_dim: int = kwargs.get("node_dim", 128)
        node_irreps: str = kwargs.get("node_irreps", "128x0e + 64x1o + 32x2e")
        embed_basis: str = kwargs.get("embed_basis", "gfn2-xtb")
        aux_basis: str = kwargs.get("aux_basis", "aux56")
        num_basis: int = kwargs.get("num_basis", 20)
        rbf_kernel: str = kwargs.get("rbf_kernel", "bessel")
        cutoff: float = kwargs.get("cutoff", 5.0)
        cutoff_fn: str = kwargs.get("cutoff_fn", "cosine")
        action_blocks: int = kwargs.get("action_blocks", 3)
        activation: str = kwargs.get("activation", "silu")
        layer_norm: bool = kwargs.get("layer_norm", True)
        charge_embed: bool = kwargs.get("charge_embed", False)
        spin_embed: bool = kwargs.get("spin_embed", False)
        output_modes: Union[str, List[str]] = kwargs.get("output_modes", ["energy"])

        self.cutoff_radius = cutoff
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
        self.mods["embedding"] = embed

        if charge_embed:
            ce = ChargeEmbedding(
                node_dim=node_dim,
                activation=activation,
            )
            self.mods["charge_embedding"] = ce
        if spin_embed:
            se = SpinEmbedding(
                node_dim=node_dim,
                activation=activation,
            )
            self.mods["spin_embedding"] = se

        for i in range(action_blocks):
            message = XPainnMessage(
                node_dim=node_dim,
                node_irreps=node_irreps,
                num_basis=num_basis,
                activation=activation,
                layer_norm=layer_norm,
            )
            update = XPainnUpdate(
                node_dim=node_dim,
                node_irreps=node_irreps,
                activation=activation,
                layer_norm=layer_norm,
            )
            self.mods[f"message_{i}"] = message
            self.mods[f"update_{i}"] = update

        if output_modes is None:
            output_modes = ["energy"]
        elif not isinstance(output_modes, Iterable):
            output_modes = [output_modes]
        for mode in output_modes:
            output = resolve_output(mode, **kwargs)
            self.mods[f"output_{mode}"] = output
            self.extra_properties.extend(output.extra_properties)


class XPaiNNEwald(XPaiNN):
    """
    XPaiNN with Ewald message passing.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        node_dim: int = kwargs.get("node_dim", 128)
        activation: str = kwargs.get("activation", "silu")
        layer_norm: bool = kwargs.get("layer_norm", True)
        use_pbc: bool = kwargs.get("use_pbc", True)
        projection_dim: int = kwargs.get("projection_dim", 8)
        ewald_blocks: int = kwargs.get("ewald_blocks", 1)
        ewald_output_modes: Union[str, List[str]] = kwargs.get(
            "ewald_output_modes", ["energy"]
        )

        if use_pbc:
            num_k_points: List[int] = kwargs.get("num_k_points", [3, 3, 3])
            ewald_initial = EwaldInitialPBC(
                num_k_points=num_k_points,
                projection_dim=projection_dim,
            )
        else:
            k_cutoff: float = kwargs.get("k_cutoff", 0.4)
            delta_k: float = kwargs.get("delta_k", 0.2)
            num_k_basis: int = kwargs.get("num_k_basis", 20)
            k_offset: Optional[float] = kwargs.get("k_offset", None)
            ewald_initial = EwaldInitialNonPBC(
                k_cutoff=k_cutoff,
                delta_k=delta_k,
                num_k_basis=num_k_basis,
                k_offset=k_offset,
                projection_dim=projection_dim,
            )
        self.mods["ewald_initial"] = ewald_initial
        for i in range(ewald_blocks):
            ewald_block = EwaldBlock(
                node_dim=node_dim,
                projection_dim=projection_dim,
                activation=activation,
                layer_norm=layer_norm,
            )
            self.mods[f"ewald_{i}"] = ewald_block
        if ewald_output_modes is None:
            ewald_output_modes = ["energy"]
        elif not isinstance(ewald_output_modes, Iterable):
            ewald_output_modes = [ewald_output_modes]
        for mode in ewald_output_modes:
            output = resolve_output(mode, **kwargs)
            self.mods[f"ewald_output_{mode}"] = output
            self.extra_properties.extend(output.extra_properties)


def resolve_model(model_name: str, **kwargs) -> BaseModel:
    models_factory = {
        "xpainn": XPaiNN,
        "xpainn-ewald": XPaiNNEwald,
    }
    if model_name.lower() not in models_factory:
        raise NotImplementedError(f"Unsupported model {model_name}")
    return models_factory[model_name.lower()](**kwargs)
