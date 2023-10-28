import argparse

import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

from xequinet.nn import XPaiNN
from xequinet.utils import (
    NetConfig,
    unit_conversion, get_default_unit, set_default_unit,
    get_atomic_energy,
)

class JitModel(XPaiNN):
    def __init__(self, config: NetConfig):
        super().__init__(config)
        self.prop_unit, self.len_unit = get_default_unit()
        atom_sp = get_atomic_energy(config.atom_ref)
        if config.batom_ref is not None:
            atom_sp -= get_atomic_energy(config.batom_ref)
        self.register_buffer("atom_sp", atom_sp)
        self.len_unit_conv = unit_conversion("Angstrom", self.len_unit)
        self.prop_unit_conv = unit_conversion(self.prop_unit, "AU")
        self.cutoff = config.cutoff
        self.max_edges = config.max_edges

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        batch: torch.LongTensor,
    ):
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
    

class JitGradModel(XPaiNN):
    def __init__(self, config: NetConfig):
        super().__init__(config)
        self.prop_unit, self.len_unit = get_default_unit()
        atom_sp = get_atomic_energy(config.atom_ref)
        if config.batom_ref is not None:
            atom_sp -= get_atomic_energy(config.batom_ref)
        self.register_buffer("atom_sp", atom_sp)
        self.len_unit_conv = unit_conversion("Angstrom", self.len_unit)
        self.prop_unit_conv = unit_conversion(self.prop_unit, "AU")
        self.grad_unit_conv = unit_conversion(f"{self.prop_unit}/{self.len_unit}", "AU")
        self.cutoff = config.cutoff
        self.max_edges = config.max_edges

    def forward(
        self,
        at_no: torch.LongTensor,
        pos: torch.Tensor,
        batch: torch.LongTensor,
    ):
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
        

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Just in time script for XequiNet")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Whether testing force additionally when the output mode is 'scalar'",
    )
    args = parser.parse_args()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    config = NetConfig.parse_obj(ckpt["config"])
    
    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    # adjust some configurations
    config.node_mean = 0.0; config.graph_mean = 0.0
    if args.force and config.output_mode == "scalar":
        config.output_mode = "grad"
    
    # build model
    if config.output_mode == "grad":
        model = JitGradModel(config).to(device)
    else:
        model = JitModel(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model_script = torch.jit.script(model)
    output_file = f"{args.ckpt.split('/')[-1].split('.')[0]}.jit"
    model_script.save(output_file)


if __name__ == "__main__":
    main()