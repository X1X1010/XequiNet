import torch 
import torch.nn as nn
from e3nn.o3 import Irreps 
from typing import List, Tuple, Union, Iterable, Dict 
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter

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
        num_reps_1d, offset_per_l = 0, [0]
        for mul, ir in self.irreps:
            cur_num_reps = mul * ir.dim 
            num_reps_1d += cur_num_reps
            offset_per_l.append(cur_num_reps + offset_per_l[-1])
        self.num_reps_1d = num_reps_1d
        for ele in possible_elements:
            atm_num = ELEMENTS_DICT[ele]
            repid_map = []
            basis_set = load_basis(basisname, ELEMENTS_LIST[atm_num])
            offset, cur_l = 0, basis_set[0][0]
            for ao in basis_set:
                l = ao[0]
                if l > cur_l:  # new shell 
                    offset = 0
                    cur_l = l 
                repid_map.append(torch.LongTensor(m_idx_map[l]) + offset + offset_per_l[l])
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
        irreps_out:Union[str, Irreps, Iterable], 
        possible_elements:List[str], 
        basisname:str = "def2svp"
    ):
        """
        Args:
            out_irreps: e3nn layout of the 2d tensor being masked.
            possible_elements: list of existing elements in the dataset.
            basisname: target basis set. 
        """
        super().__init__() 
        self.irreps_out = irreps_out if isinstance(irreps_out, Irreps) else Irreps(irreps_out)
        self.num_channels_1d = self.irreps_out.num_irreps
        num_reps_1d, offset_per_l = 0, [0]
        for mul, ir in self.irreps_out:
            cur_num_reps = mul * ir.dim
            num_reps_1d += cur_num_reps
            offset_per_l.append(cur_num_reps + offset_per_l[-1])
        self.num_reps_1d = num_reps_1d
        out_repid_mask = torch.zeros(120, self.num_reps_1d, dtype=torch.int32)

        for ele in possible_elements:
            atm_num = ELEMENTS_DICT[ele]
            basis_set = load_basis(basisname, ELEMENTS_LIST[atm_num])
            offset, cur_l = 0, basis_set[0][0]
            for ao in basis_set:
                l = ao[0]
                if l > cur_l:   # new shell 
                    offset = 0 
                    cur_l = l 
                pos_idx = torch.arange(2*l+1) + offset + offset_per_l[l]
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


def cal_orbital_and_energies(overlap_matrix:torch.Tensor, full_hamiltonian:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert overlap_matrix.shape[0] == full_hamiltonian.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eps = 1e-8 * torch.ones_like(eigvals)
    eigvals = torch.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

    Fs = torch.bmm(torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients


class QCMatriceBuilder(nn.Module):
    def __init__(
        self, 
        irreps_out:Union[str, Irreps, Iterable], 
        possible_elements:List[str],
        basisname:str="def2svp",
        m_idx_map=m_idx_map,
    ):
        super().__init__() 
        self.irreps_out = irreps_out if isinstance(irreps_out, Irreps) else Irreps(irreps_out) 
        self.num_channels_1d = self.irreps_out.num_irreps
        out_repid_map = torch.zeros(self.num_reps_1d, dtype=torch.long)
        elem_num_basis = torch.zeros(120, 1, dtype=torch.int32)
        # out_repid_map 
        num_reps_1d, src_id, offset_per_l = 0, [0], 0
        for mul, ir in self.irreps_out:
            cur_num_reps = mul * ir.dim
            num_reps_1d += cur_num_reps
            offset_per_l.append(cur_num_reps + offset_per_l[-1])
            offset = 0
            for dst_n_current in range(mul):
                for m_qc in range(2*ir.l + 1):
                    m_out = m_idx_map[ir.l][m_qc] 
                    out_repid_map[src_id] = offset_per_l[ir.l] + offset + m_out 
                    offset += 2*ir.l + 1
                    src_id += 1 
        self.num_reps_1d = num_reps_1d
        # num_basis for each element 
        for ele in possible_elements:
            atm_num = ELEMENTS_DICT[ele]
            basis_set = load_basis(basisname, ELEMENTS_LIST[atm_num])
            for ao in basis_set:
                l = ao[0]
                elem_num_basis[ele] += 2*l + 1 
        # register buffer
        self.register_buffer("elem_num_basis", elem_num_basis)
        self.register_buffer("out_repid_map", out_repid_map)

    def forward(
            self, 
            res_ten:Tuple[torch.Tensor, torch.Tensor], 
            raw_mask:Tuple[torch.BoolTensor, torch.BoolTensor], 
            atomic_numbers:torch.Tensor, 
            edge_index:torch.Tensor, 
            batch_index:torch.Tensor, 
            natms:torch.LongTensor
        ):
        # transform e3nn's rep layout to qc interleaved layout along the two dims where the matrice is stored.
        # dimension -1
        node_ten_tmp = torch.index_select(res_ten[0], dim=2, index=self.out_repid_map)
        edge_ten_tmp = torch.index_select(res_ten[1], dim=2, index=self.out_repid_map)
        node_mask_tmp = torch.index_select(raw_mask[0], dim=2, index=self.out_repid_map) 
        edge_mask_tmp = torch.index_select(raw_mask[1], dim=2, index=self.out_repid_map) 
        # dimension -2 
        node_ten = torch.index_select(node_ten_tmp, dim=1, index=self.out_repid_map)
        edge_ten = torch.index_select(edge_ten_tmp, dim=1, index=self.out_repid_map)
        node_mask = torch.index_select(node_mask_tmp, dim=1, index=self.out_repid_map)
        edge_mask = torch.index_select(edge_mask_tmp, dim=1, index=self.out_repid_map)
        # create a huge tmp tensor to hold batched result 
        num_mole, max_natms = natms.shape[0], torch.max(natms).item()
        res = torch.zeros(num_mole, max_natms, max_natms, self.num_reps_1d, self.num_reps_1d, device=node_ten.device, dtype=node_ten.dtype)
        res_mask = torch.zeros(num_mole, max_natms, max_natms, self.num_reps_1d, self.num_reps_1d, device=node_ten.device, dtype=torch.bool)
        # get the huge mask 
        diagnol_mask = torch.eye(max_natms, device=res.device).expand(*res.shape[:3]).bool()
        mask1d_list = [] 
        for idx in range(num_mole):
            mask1d = torch.zeros(max_natms, dtype=torch.bool, device=res.device)
            mask1d[:natms[idx]] = True
            mask1d_list.append(mask1d) 
        diagnol_mask1d = torch.cat(mask1d_list) 
        res[diagnol_mask, :, :][diagnol_mask1d, :, :] = node_ten 
        res_mask[diagnol_mask, :, :][diagnol_mask1d, :, :] = node_mask
        # scatter edge sub-blocks onto off-diagnol part of the res matrix 
        off_diagnol_mask = to_dense_adj(edge_index, batch=batch_index, max_atoms=max_natms).bool()
        res[off_diagnol_mask, :, :] = edge_ten 
        res_mask[off_diagnol_mask, :, :] = edge_mask

        elem_num_basis = self.elem_num_basis[atomic_numbers, :]
        res_slices = scatter(elem_num_basis, batch_index, dim=0)

        return res[res_mask], res_slices


class BuildMatPerMole(nn.Module):
    def __init__(
        self, 
        irreps_out:Union[str, Irreps, Iterable], 
        possible_elements:List[str],
        basisname:str="def2svp",
        m_idx_map=m_idx_map,
    ):
        super().__init__() 
        self.irreps_out = irreps_out if isinstance(irreps_out, Irreps) else Irreps(irreps_out) 
        self.num_channels_1d = self.irreps_out.num_irreps
        out_repid_map = torch.zeros(self.num_reps_1d, dtype=torch.long)
        elem_num_basis = torch.zeros(120, 1, dtype=torch.int32)
        # out_repid_map 
        num_reps_1d, src_id, offset_per_l = 0, [0], 0
        for mul, ir in self.irreps_out:
            cur_num_reps = mul * ir.dim
            num_reps_1d += cur_num_reps
            offset_per_l.append(cur_num_reps + offset_per_l[-1])
            offset = 0
            for dst_n_current in range(mul):
                for m_qc in range(2*ir.l + 1):
                    m_out = m_idx_map[ir.l][m_qc] 
                    out_repid_map[src_id] = offset_per_l[ir.l] + offset + m_out 
                    offset += 2*ir.l + 1
                    src_id += 1 
        self.num_reps_1d = num_reps_1d
        # num_basis for each element 
        for ele in possible_elements:
            atm_num = ELEMENTS_DICT[ele]
            basis_set = load_basis(basisname, ELEMENTS_LIST[atm_num])
            for ao in basis_set:
                l = ao[0]
                elem_num_basis[ele] += 2*l + 1 
        # register buffer
        self.register_buffer("elem_num_basis", elem_num_basis)
        self.register_buffer("out_repid_map", out_repid_map)

    def forward(
            self, 
            res_node:torch.Tensor,
            res_edge:torch.Tensor,
            raw_node_mask:torch.BoolTensor,
            raw_edge_mask:torch.BoolTensor,
            atomic_numbers:torch.Tensor, 
            edge_index:torch.Tensor, 
            # batch_index:torch.LongTensor,
        ):
        # transform e3nn's rep layout to qc interleaved layout along the two dims where the matrice is stored.
        # dimension -1
        node_ten_tmp = torch.index_select(res_node, dim=2, index=self.out_repid_map)
        edge_ten_tmp = torch.index_select(res_edge, dim=2, index=self.out_repid_map)
        node_mask_tmp = torch.index_select(raw_node_mask, dim=2, index=self.out_repid_map) 
        edge_mask_tmp = torch.index_select(raw_edge_mask, dim=2, index=self.out_repid_map) 
        # dimension -2 
        node_ten = torch.index_select(node_ten_tmp, dim=1, index=self.out_repid_map)
        edge_ten = torch.index_select(edge_ten_tmp, dim=1, index=self.out_repid_map)
        node_mask = torch.index_select(node_mask_tmp, dim=1, index=self.out_repid_map)
        edge_mask = torch.index_select(edge_mask_tmp, dim=1, index=self.out_repid_map)
        # create a sratch tensor to hold output 
        natms = atomic_numbers.shape[0]
        res = torch.zeros(natms, natms, self.num_reps_1d, self.num_reps_1d, device=res_node.device, dtype=res_node.dtype)
        res_mask = torch.zeros(natms, natms, self.num_reps_1d, self.num_reps_1d, device=res_node.device, dtype=torch.bool)
        # get the huge mask 
        diagnol_mask = torch.eye(natms, device=res.device).bool()
        
        res[diagnol_mask, :, :] = node_ten 
        res_mask[diagnol_mask, :, :] = node_mask
        # scatter edge sub-blocks onto off-diagnol part of the res matrix 
        batch_index = torch.zeros(natms, device=res.device, dtype=torch.long)
        off_diagnol_mask = to_dense_adj(edge_index, batch=batch_index, max_atoms=natms).bool()
        res[off_diagnol_mask, :, :] = edge_ten 
        res_mask[off_diagnol_mask, :, :] = edge_mask
        tot_num_basis = torch.sum(self.elem_num_basis[atomic_numbers]).item()
        return res[res_mask].reshape(tot_num_basis, tot_num_basis)

