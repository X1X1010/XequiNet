import torch 
import torch.nn as nn
from e3nn.o3 import Irreps 
from typing import List, Tuple, Union, Iterable, Dict 

from .qc import ELEMENTS_DICT, ELEMENTS_LIST, load_basis 


m_idx_map = {
    0: [0],
    1: [2, 0, 1],  # (x, y, z) to (y, z, x)
    2: [0, 1, 2, 3, 4],
    3: [0, 1, 2, 3, 4, 5, 6],
    4: [0, 1, 2, 3, 4, 5, 6, 7, 8],
}


class TwoBodyBlockPad(nn.Module):
    r"""
    Class for converting 2-body feature (order 2 tensor) from QC calculation to e3nn format 2d layout.
    The returned 2d SO(3) tensor is of symmetric layout along the rep_dims. 
    """
    def __init__(
        self,
        irreps_out: Union[str, Irreps, Iterable],
        rep_dims: Tuple[int, int],
        possible_elements: List[str],
        basisname: str = "def2svp",
        m_idx_map: Dict[int, List[int]] = m_idx_map
    ):
        """
        Args:
            irreps_out (Irreps): e3nn.o3.Irreps, layout of the output padded 2d tensor.
            rep_dims (tuple): original two dimensions of the two-body feature tensor.
            possible_elements (list): list of exisisting elements in the dataset.
            basisname (str): name of the target basis set.
            m_idx_map (dict): mapping from output m index from the QC program to torch_gauge's m index. 
        """
        super().__init__()
        self.irreps = irreps_out if isinstance(irreps_out, Irreps) else Irreps(irreps_out)
        self.rep_dims = rep_dims
        self._generate_buffers(possible_elements, basisname, m_idx_map) 

    def _generate_buffers(self, possible_elements, basisname, m_idx_map):
        out_repid_map = {}
        self.num_channels_1d = self.irreps.num_irreps 
        num_reps_1d = 0
        for mul, ir in self.irreps:
            num_reps_1d += mul * ir.dim
        self.num_reps_1d = num_reps_1d
        for ele in possible_elements:
            atm_num = ELEMENTS_DICT[ele]
            repid_map = []
            basis_set = load_basis(basisname, ELEMENTS_LIST[atm_num])
            offset = 0
            for ao in basis_set:
                l = ao[0]
                repid_map.append(torch.LongTensor(m_idx_map[l]) + offset)
                offset += 2 * l + 1
            repid_map = torch.cat(repid_map)
            out_repid_map[ele] = torch.nn.Parameter(repid_map, requires_grad=False) 
        self.out_repid_map = torch.nn.ParameterDict(out_repid_map) 
    
    def forward(self, atomsybs:Iterable, feat_ten:torch.Tensor) -> torch.Tensor:
        """
        Args:
            atomsybs (Iterable): atoms in the current molecule.
            feat_ten (torch.Tensor): input 2-body feature tensor. 
        Return:
           torch.Tensor: padded tensor of shape (natm, natm, Irrep, Irrep), natm refers to number of atoms in the system.
                As input matrice is a fully connected graph by abstract.
        """
        # generate dst idx for padding 
        dst_rep_ids_1d = torch.cat([self.out_repid_map[ele] for ele in atomsybs]) 
        dst_offsets_1d = torch.arange(
            len(atomsybs), dtype=torch.long, device=feat_ten.device
        ).repeat_interleave(
            torch.tensor([len(self.out_repid_map[ele]) for ele in atomsybs], dtype=torch.long, device=feat_ten.device) 
        )
        dst_flat_ids_1d = dst_offsets_1d * self.num_reps_1d + dst_rep_ids_1d 
        # prepare for output 
        sp_2body_feat_ten_flat = torch.zeros(
            *feat_ten.shape[: self.rep_dims[0]], (self.num_reps_1d * len(atomsybs)) ** 2, *feat_ten.shape[self.rep_dims[1] + 1 :],
            dtype = feat_ten.dtype,
            device = feat_ten.device
        )
        # scatter interleaved reps to the padded 2d layout
        dst_flat_ids_2d = (
            dst_flat_ids_1d.unsqueeze(1) * (self.num_reps_1d * len(atomsybs)) + dst_flat_ids_1d.unsqueeze(0)
        ).view(-1) 
        sp_2body_feat_ten_flat.index_add_(self.rep_dims[0], dst_flat_ids_2d, feat_ten.flatten(*self.rep_dims)) 
        sp_2body_feat_ten = (
            sp_2body_feat_ten_flat.view(
                *feat_ten.shape[:self.rep_dims[0]], 
                len(atomsybs), 
                self.num_reps_1d, 
                len(atomsybs), 
                self.num_reps_1d,
                *feat_ten.shape[self.rep_dims[1]+1 :] 
            ).transpose(self.rep_dims[1], self.rep_dims[1] + 1)
            .contiguous()
        )
        return sp_2body_feat_ten


class TwoBodyBlockMask(nn.Module):
    r'''
    module to generate mask for two body feature Irrep padded tensor for batched training.
    '''
    def __init__(
        self, 
        out_irreps:Union[str, Irreps, Iterable], 
        possible_elements:List[int], 
        basisname:str = "def2-tzvp"
    ):
        """
        Args:
            out_irreps: e3nn layout of the 2d tensor being masked.
            possible_elements: list of existing elements in the dataset.
            basisname: target basis set. 
        """
        super().__init__() 
        self.out_irreps = out_irreps if isinstance(out_irreps, Irreps) else Irreps(out_irreps)
        self.num_channels_1d = self.out_irreps.num_irreps
        num_reps_1d = 0
        for mul, ir in self.out_irreps:
            num_reps_1d += mul * ir.dim
        self.num_reps_1d = num_reps_1d
        out_repid_mask = torch.zeros(120, self.num_reps_1d, dtype=torch.int32)

        for ele in possible_elements:
            atm_num = ELEMENTS_DICT[ele]
            basis_set = load_basis(basisname, ELEMENTS_LIST[atm_num])
            offset = 0
            for ao in basis_set:
                l = ao[0]
                pos_idx = torch.arange(2*l+1) + offset 
                out_repid_mask[atm_num][pos_idx] = 1
                offset += 2*l + 1
        self.register_buffer("out_repid_mask", out_repid_mask.bool())

    def forward(self, atomic_numbers:torch.Tensor, edge_index:torch.Tensor) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        Args:
            atomic_numbers(torch.Tensor): list of atomic numbers in the current batch.
            edge_index(torch.Tensor): PyG edge index of shape (2, num_edge).
        Return:
            mask for node and edge tensor 
        """
        dst_rep_1d_mask = self.out_repid_mask[atomic_numbers, :]
        # Diagnol mask for node 
        dst_rep_node_mask = dst_rep_1d_mask.unsqueeze(1) * dst_rep_1d_mask.unsqueeze(2) 
        # Off-diagnol mask for edge 
        dst_rep_ket_mask_1d = torch.index_select(dst_rep_1d_mask, dim=0, index=edge_index[1]).view(edge_index.shape[1], 1, self.num_reps_1d)
        dst_rep_bra_mask_1d = torch.index_select(dst_rep_1d_mask, dim=0, index=edge_index[0]).view(edge_index.shape[1], self.num_reps_1d, 1) 
        dst_rep_edge_mask = (dst_rep_bra_mask_1d * dst_rep_ket_mask_1d)
        return dst_rep_node_mask.bool(), dst_rep_edge_mask.bool()
    

class Mat2GraphLabel(nn.Module):
    r"""
    module to convert QC matrice to PyG data label for each atom (node) and 
    atom pair (edge of fully connected graph).
    """
    def __init__(
        self, 
        target_irreps:Union[str, Irreps, Iterable], 
        possible_elements:List[int], 
        basisname:str="def2tzvp"
    ):
        """
        Args:
            target_irreps: e3nn layout of the 2d tensor being masked.
            possible_elements: list of existing elements in the dataset.
            basisname: target basis set. 
        """
        super().__init__()
        target_irreps = target_irreps if isinstance(target_irreps, Irreps) else Irreps(target_irreps)
        self.twobodypad = TwoBodyBlockPad(
            target_irreps,
            rep_dims=(0, 1),
            possible_elements=possible_elements,
            basisname=basisname,
        )

    def forward(self, data, feat_matrice:torch.Tensor, atom_sybs:Iterable=None, edge_index:torch.Tensor=None):
        num_nodes = len(data.at_no)
        folded_X = self.twobodypad(atom_sybs, feat_matrice)
        
        diagnol_mask = torch.eye(num_nodes, dtype=torch.bool, device=feat_matrice.device) 
        node_label = folded_X[diagnol_mask, :, :] 
        if edge_index is None:
            # as a fully connected graph 
            edge_mask = torch.logical_not(diagnol_mask) 
            edge_index = torch.nonzero(edge_mask).T.long() 
            setattr(data, "fc_edge_index", edge_index)
        edge_label = folded_X[edge_index[0], edge_index[1], ...] 
        return node_label, edge_label


