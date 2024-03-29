from typing import Tuple
from functools import reduce
from pyscf import gto, lib, symm, tddft, dft
from pyscf.scf import hf_symm, hf
import numpy as np
from scipy.linalg import sqrtm, eigh

import torch
from torch_cluster import radius_graph
from ..nn import resolve_model
from ..data import TextDataset
from ..utils import set_default_unit, unit_conversion, NetConfig, MatToolkit


def Lowdin_eri(mf: hf.SCF) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = {
        'H':  6.429954422, "He": 12.54491189, "Li": 2.374586656, "Be": 3.496763353, 'B':  4.619008972,
        'C':  5.740978922, 'N':  6.862466529, 'O':  7.985435701, 'F':  9.106475372, "Ne": 10.2303405,
        "Na": 2.44414136,  "Mg": 3.014651383, "Al": 3.584907074, "Si": 4.1551309,   'P':  4.725803974,
        'S':  5.295979241, "Cl": 5.866186484, "Ar": 6.436618714, 'K':  2.327317836, "Ca": 2.758723814,
        "Sc": 2.858192114, "Ti": 2.957830043, 'V':  3.057341006, "Cr": 3.156725429, "Mn": 3.256382723,
        "Fe": 3.355931405, "Co": 3.455609117, "Ni": 3.555013313, "Cu": 3.654418348, "Zn": 3.754160145,
        "Ga": 4.185519793, "Ge": 4.616627246, "As": 5.066214507, "Se": 5.479496097, "Br": 5.911099645,
        "Kr": 6.341846768, "Rb": 2.120458257, "Sr": 2.537370048, 'Y':  2.633546898, "Zr": 2.729752893,
        "Nb": 2.825973886, "Mo": 2.922129604, "Tc": 3.018370878, "Ru": 3.114598177, "Rh": 3.21075628,
        "Pd": 3.306947448, "Ag": 3.403194857, "Cd": 3.499376139, "In": 3.916369246, "Sn": 4.333233219,
        "Sb": 4.750078786, "Te": 5.166979327, 'I':  5.583887102, "Xe": 6.00089733,  "Cs": 0.682915024,
        "Ba": 0.920094684, "La": 1.157088786, "Ce": 1.3942757,   "Pr": 1.631473173, "Nd": 1.868438998,
        "Pm": 2.105657793, "Sm": 2.342664642, "Eu": 2.579814982, "Gd": 2.817026423, "Tb": 3.054036533,
        "Dy": 3.291169231, "Ho": 3.528297161, "Er": 3.765524929, "Tm": 4.002554703, "Yb": 4.239478341,
        "Lu": 4.476583021, "Hf": 4.706522449, "Ta": 4.950846694, 'W':  5.187931172, "Re": 5.425607621,
        "Os": 5.661914431, "Ir": 5.90004292,  "Pt": 6.136714532, "Au": 6.374129977, "Hg": 6.610265613,
        "Tl": 1.704348581, "Pb": 1.941352612, "Bi": 2.17849151,  "Po": 2.415812106, "At": 2.652778084,
        "Rn": 2.889955457, "Fr": 0.988252988, "Ra": 1.281949997, "Ac": 1.349725038,
    }
    mol = mf.mol

    s = mol.intor_symmetric('int1e_ovlp')
    s_half = np.real(sqrtm(s))
    mo_coeff_ortho = s_half @ mf.mo_coeff

    q_a_pq = np.zeros((mol.natm, mol.nao, mol.nao))
    c_mp_c_mq = lib.einsum("mp, mq -> mpq", mo_coeff_ortho, mo_coeff_ortho)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        q_a_pq[s[0]] += c_mp_c_mq[i]
    ni = mf._numint
    if mf.xc == "CAM-B3LYP":
        alpha = 0.90  # The alpha in reference is beta in the code.
        beta = 1.86 
        hyb = 0.38
    elif mf.xc == "LC-BLYP":
        alpha = 4.50
        beta = 8.00
        hyb = 0.53
    elif mf.xc == "wB97":
        alpha = 4.41
        beta = 8.00
        hyb = 0.61
    elif mf.xc == "wB97X":
        alpha = 4.58
        beta = 8.00
        hyb = 0.56
    else:
        hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)[2]
        alpha0 = 1.42
        alpha1 = 0.48
        beta0 = 0.2
        beta1= 1.83
        alpha = alpha0 + hyb * alpha1
        beta = beta0 + hyb * beta1
    R_ab = gto.inter_distance(mol)
    eta_ab = np.zeros((mol.natm, mol.natm))
    for i, a in enumerate(mol._atom):
        for j, b in enumerate(mol._atom):
            eta_ab[i][j] = (eta[a[0]] + eta[b[0]]) / 2
    eta_ab *= 2 * unit_conversion("eV", "Hartree")
    g_ab_j = 1 / ((R_ab**beta + (hyb * eta_ab)**(-beta))**(1/beta))
    g_ab_k = 1 / ((R_ab**alpha + eta_ab**(-alpha))**(1/alpha))
    q_b_rs_g_ab_j = lib.einsum("brs, ab -> ars", q_a_pq, g_ab_j)
    q_b_rs_g_ab_k = lib.einsum("brs, ab -> ars", q_a_pq, g_ab_k)
    return q_a_pq, q_b_rs_g_ab_j, q_b_rs_g_ab_k


def A_X_build(mf, singlet=True):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = np.where(mo_occ==2)[0]
    orbo = mo_coeff[:, occidx]
    nocc = orbo.shape[1]
    q_a_pq, q_b_rs_g_ab_j, q_b_rs_g_ab_k = Lowdin_eri(mf)
    q_a_ij = q_a_pq[:, :nocc, :nocc]
    q_b_ab_g_ab_j = q_b_rs_g_ab_j[:, nocc:, nocc:]

    if singlet == True:
        q_a_ia = q_a_pq[:, :nocc, nocc:]
        q_b_jb_g_ab_k = q_b_rs_g_ab_k[:, :nocc, nocc:]

        def A_X(X: np.ndarray) -> np.ndarray:
            result = 2 * np.einsum("Aia, Ajb, xjb -> xia", q_a_ia, q_b_jb_g_ab_k, X, optimize=True)
            for x in range(X.shape[0]):
                result[x] -= np.einsum("Aij, Aab, jb -> ia", q_a_ij, q_b_ab_g_ab_j ,X[x], optimize=True)
            return result
    else:
        def A_X(X: np.ndarray) -> np.ndarray:
            result = -1 * np.einsum("Aij, Aab, xjb -> xia", q_a_ij, q_b_ab_g_ab_j, X, optimize=True)
            return result
        
    return A_X    


def gen_stda_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute A x

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    # print('mo_coeff', mo_coeff.shape, mo_coeff.dtype)
    assert (mo_coeff.dtype == np.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidx = np.where(mo_occ==2)[0]
    viridx = np.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        orbsym_in_d2h = np.asarray(orbsym) % 10  # convert to D2h irreps
        sym_forbid = (orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym

    if fock_ao is None:
        #dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        #fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
        foo = np.diag(mo_energy[occidx])
        fvv = np.diag(mo_energy[viridx])
    else:
        fock = reduce(np.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))   # here replace fock_ao with fock_matrix
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]

    hdiag = fvv.diagonal() - foo.diagonal()[:,None]
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()

    mo_coeff = np.asarray(np.hstack((orbo,orbv)), order='F')
    #vresp = mf.gen_response(singlet=singlet, hermi=0)
    vresp = A_X_build(mf, singlet=singlet)

    def vind(zs):
        zs = np.asarray(zs).reshape(-1,nocc,nvir)
        if wfnsym is not None and mol.symmetry:
            zs = np.copy(zs)
            zs[:,sym_forbid] = 0

        # *2 for double occupancy
        #dmov = lib.einsum('xov,qv,po->xpq', zs*2, orbv.conj(), orbo)
        v1ov = vresp(zs)
        #v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)
        v1ov += lib.einsum("xqs,sp->xqp", zs, fvv)
        v1ov -= lib.einsum("xpr,sp->xsr", zs, foo)
        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
        return v1ov.reshape(v1ov.shape[0],-1)

    return vind, hdiag


class sTDA(tddft.rhf.TDA):

    def gen_vind(self, mf=None):
        '''Generate function to compute Ax'''
        if mf is None:
            mf = self._scf
        return gen_stda_operation(mf, singlet=self.singlet, wfnsym=self.wfnsym)
        

def cal_orbital_and_energies(ovlp: np.ndarray, fock: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    orb_energies, orb_coeff = eigh(fock, ovlp)
    idx = np.argmax(abs(orb_coeff.real), axis=0)
    orb_coeff[:,orb_coeff[idx,np.arange(len(orb_energies))].real<0] *= -1
    return orb_energies, orb_coeff


def run_std_from_fock(args) -> None:

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    config = NetConfig.model_validate(ckpt["config"])

    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # load input data
    dataset = TextDataset(args.input)
    outp = f"{args.input.split('/')[-1].split('.')[0]}.log" if args.output is None else args.output


    with open(outp, 'w') as wf:
        wf.write("XequiNet predict Fock matrix and run sTD\n")
    
    mat_toolkit = MatToolkit(config.target_basis, config.possible_elements)
    for imol, data in enumerate(dataset, start=1):
        with torch.no_grad():
            data = data.to(device)
            data.batch = torch.zeros_like(data.at_no, dtype=torch.long, device=device)
            data.edge_index = radius_graph(data.pos, r=config.cutoff, max_num_neighbors=config.max_edges)
            if config.require_full_edges:
                data.edge_index_full = mat_toolkit.get_edge_index_full(data.at_no)
                mat_edge_index = data.edge_index_full
            else:
                mat_edge_index = data.edge_index
            node_padded, edge_padded = model(data)
            fock = mat_toolkit.assemble_blocks(
                data.at_no, node_padded, edge_padded, mat_edge_index
            ).cpu().numpy() * unit_conversion(config.default_property_unit, "Hartree")

        at_no = data.at_no.cpu().numpy()
        coord = data.pos.cpu().numpy()
        charge = data.charge.to(torch.long).item()
        mol = gto.Mole()
        t = [(a, c) for a, c in zip(at_no, coord)]
        mol.build(
            atom=t,
            charge=charge,
            basis=config.target_basis,
            unit=config.default_length_unit,
            verbose=args.verbose,
            output=outp,
        )
    
        ovlp = mol.intor("int1e_ovlp")
        orb_energies, orb_coeff = cal_orbital_and_energies(ovlp, fock)

        occ = (at_no.sum() - charge) // 2
        nao = mol.nao  # total numbers of orbitals
        mo_occ = [2] * occ + [0] * (nao - occ) 
        mo_occ = np.array(mo_occ) 

        fake_method = dft.RKS(mol)
        fake_method.verbose = args.verbose
        fake_method.xc = "B3LYP"
        fake_method.mo_occ = mo_occ
        fake_method.mo_energy = orb_energies
        fake_method.mo_coeff = orb_coeff
        fake_method.converged = True

        stda = sTDA(fake_method)
        e, xy = stda.kernel(nstates=args.nstates)
        e_eV = e * unit_conversion("Hartree", "eV")

        with open(outp, 'a') as wf:
            wf.write(f"sTD result for molecule {imol:6d}: \n")
            for i, excite_eng in enumerate(e_eV):
                wf.write(f"Excited state {i + 1}: {excite_eng:10.4f}\n")

