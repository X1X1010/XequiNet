from typing import Tuple, Union

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_scatter import scatter

from xequinet.utils import NetConfig

from .model import XPaiNN, PBCPaiNN
from ..utils import NetConfig, unit_conversion, get_default_unit, get_atomic_energy


class BaseJitModel(nn.Module):
    def __init__(self, config: NetConfig):
        self.prop_unit, self.len_unit = get_default_unit()
        atom_sp = get_atomic_energy(config.atom_ref) - get_atomic_energy(config.batom_ref)
        self.register_buffer("atom_sp", atom_sp)
        self.len_unit_conv = unit_conversion("Angstrom", self.len_unit)
        self.prop_unit_conv = unit_conversion(self.prop_unit, "AU")
        self.cutoff = config.cutoff
        self.max_edges = config.max_edges


class GradJitModel(BaseJitModel):
    def __init__(self, config: NetConfig):
        super().__init__(config)
        self.grad_unit_conv = unit_conversion(f"{self.prop_unit}/{self.len_unit}", "AU")


class JitXPaiNN(XPaiNN, BaseJitModel):
    def __init__(self, config: NetConfig):
        super().__init__(config)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        batch: torch.LongTensor,
    ) -> torch.Tensor:
        pos = pos * self.len_unit_conv
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_edges)
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        model_res = self.out(x_scalar, x_vector, pos, batch)
        atom_energies = self.atom_sp[at_no]
        result = model_res.double().index_add(0, batch, atom_energies)
        result = result * self.prop_unit_conv
        return result


class GradJitXPaiNN(XPaiNN, GradJitModel):
    def __init__(self, config: NetConfig):
        super().__init__(config)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        batch: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = pos * self.len_unit_conv
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_edges)
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        model_prop, model_neg_grad = self.out(x_scalar, x_vector, pos, batch)
        atom_energies = self.atom_sp[at_no]
        prop_res = model_prop.double().index_add(0, batch, atom_energies) * self.prop_unit_conv
        neg_grad = model_neg_grad.double() * self.grad_unit_conv
        return prop_res, neg_grad


class JitPBCPaiNN(PBCPaiNN, BaseJitModel):
    def __init__(self, config: NetConfig):
        super().__init__(config)

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        shifts: torch.Tensor,
        batch: torch.LongTensor,
        at_filter: torch.BoolTensor,
    ) -> torch.Tensor:
        pos = pos * self.len_unit_conv
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_edges)
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, shifts, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        model_res = self.out(x_scalar, x_vector, pos, batch, at_filter)
        atom_energies = self.atom_sp[at_no]
        result = model_res.double().index_add(0, batch, atom_energies)
        result = result * self.prop_unit_conv
        return result
    

class GradJitPBCPaiNN(PBCPaiNN, GradJitModel):
    def __init__(self, config: NetConfig):
        super().__init__(config)
    
    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        shifts: torch.Tensor,
        batch: torch.LongTensor,
        at_filter: torch.BoolTensor,
    ) -> torch.Tensor:
        pos = pos * self.len_unit_conv
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_edges)
        x_scalar, rbf, fcut, rsh = self.embed(at_no, pos, shifts, edge_index)
        x_vector = torch.zeros((x_scalar.shape[0], rsh.shape[1]), device=x_scalar.device)
        for msg, upd in zip(self.message, self.update):
            x_scalar, x_vector = msg(x_scalar, x_vector, rbf, fcut, rsh, edge_index)
            x_scalar, x_vector = upd(x_scalar, x_vector)
        model_prop, model_neg_grad = self.out(x_scalar, x_vector, pos, batch, at_filter)
        atom_energies = self.atom_sp[at_no]
        prop_res = model_prop.double().index_add(0, batch, atom_energies) * self.prop_unit_conv
        neg_grad = model_neg_grad.double() * self.grad_unit_conv
        return prop_res, neg_grad


def resolve_jit_model(config: NetConfig) -> nn.Module:
    if config.pbc:
        if config.output_mode == "grad":
            return GradJitPBCPaiNN(config)
        else:
            return JitPBCPaiNN(config)
    else:
        if config.output_mode == "grad":
            return GradJitXPaiNN(config)
        else:
            return JitXPaiNN(config)
