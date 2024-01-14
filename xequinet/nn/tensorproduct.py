'''
Utility for cutomizing e3nn.TensorProduct
'''
from e3nn.o3 import Irreps
import math 

# from QHNet https://github.com/divelab/AIRS/OpenDFT/QHBench
def prod(x):
    """
    Return product of the input sequence.
    """
    out = 1 
    for a in x:
        out *= a 
    return out 

# customized from QHNet, 
# see also https://docs.e3nn.org/en/latest/api/o3/o3_tp.html for TensorProduct
def get_feasible_tp(
    irreps_in1:Irreps, 
    irreps_in2:Irreps, 
    filter_irreps_out:Irreps, 
    tp_mode:str="uvw", 
    trainable:bool=True
):
    r"""
    Generate irreps_out and instructions for customized TensorProduct.
    Args:
        irreps_in1: irreducible representations of input tensor 1
        irreps_in2: irreducible representations of input tensor 2
        filter_irreps_out: target output irreps
        tp_mode: tp_mode for the customized TensorProduct in each path
        trainable: whether the path in TensorProduct contain trainable parameters
    Return:
        irreps_mid: the final output irreps decided from input arguments
        instructions: instructions for each TP path in customized TensorProduct
    Note:
        The returned irreps_mid is not necessarily same as filter_irreps_out. In fact,
        the layout of irreps_mid are decided on irreps_in1, irreps_in2 via selection rule
        of angular momentum, on filter_irreps_out by min and max angular momentum channels
        desired and on tp_mode for multiplicity of each irrep. 
    Tips:
        It is recommended to set same multiplicity for each l channel in filter_irreps_out
        for the sake of ease, i.e. 64x0e + 64x1e + 64x2e, etc.
    """
    assert tp_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv"]
    irreps_mid = []
    instructions = [] 
    # loop over irreps in irreps_in1
    for i, (mul_1, ir_in1) in enumerate(irreps_in1):
        # loop over irreps in irreps_in2
        for j, (mul_2, ir_in2) in enumerate(irreps_in2):
            # angular momentum selective rule 
            for ir_out in ir_in1 * ir_in2:
                if ir_out in filter_irreps_out:
                # according to tp_mode 
                    if tp_mode == "uvw":
                        mul_out = filter_irreps_out.count(ir_out)
                    elif tp_mode == "uvu":
                        mul_out = mul_1
                    elif tp_mode == "uvv":
                        mul_out = mul_2
                    elif tp_mode == "uuu":
                        assert mul_1 == mul_2 
                        mul_out = mul_1
                    elif tp_mode == "uuw":
                        assert mul_1 == mul_2 
                        mul_out = filter_irreps_out.count(ir_out)
                    elif tp_mode == "uvuv":
                        mul_out = mul_1 * mul_2 
                    else:
                        raise NotImplementedError(f"Unsupported TensorProduct constraction rule: {tp_mode}")
                    # generate tp path to kth irrep in cutoff_irreps_out 
                    if (mul_out, ir_out) not in irreps_mid: 
                        k = len(irreps_mid)
                        irreps_mid.append((mul_out, ir_out))
                    else:
                        k = irreps_mid.index((mul_out, ir_out)) 
                    instructions.append((i, j, k, tp_mode, trainable)) 
    irreps_mid = Irreps(irreps_mid) 
    normalization_coefficients = []
    for ins in instructions:
        ins_dict = {
            "uvw": irreps_in1[ins[0]].mul * irreps_in2[ins[1]].mul,
            "uvu": irreps_in2[ins[1]].mul,
            "uvv": irreps_in1[ins[0]].mul,
            "uuw": irreps_in1[ins[0]].mul,
            "uuu": 1,
            "uvuv": 1,
        }
        alpha = irreps_mid[ins[2]].ir.dim 
        x = sum([ins_dict[ins[3]] for ins in instructions])
        if x > 0.0:
            alpha /= x 
        normalization_coefficients += [math.sqrt(alpha)]

    irreps_mid, p, _ = irreps_mid.sort() 
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train, alpha)
        for (i_in1, i_in2, i_out, mode, train), alpha 
        in zip(instructions, normalization_coefficients)
    ]
    return irreps_mid, instructions 
