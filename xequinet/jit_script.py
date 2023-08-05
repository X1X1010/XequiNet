import os
import argparse

import torch

from xequinet.nn import xPaiNN
from xequinet.utils import NetConfig


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Just in time script for xPaiNN")
    parser.add_argument("--ckpt", "-c", type=str, required=True, help="Checkpoint file")
    parser.add_argument("--output-dir", "-o", type=str, default=".")
    parser.add_argument("--force", "-f", action="store_true", help="Output force")
    args = parser.parse_args()

    # load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)

    # load config
    config = NetConfig.parse_obj(ckpt["config"])
    config.node_mean = 0.0; config.graph_mean = 0.0
    if args.force:
        config.output_mode = "grad"

    # build model
    model = xPaiNN(config).to(device)
    model.load_state_dict(ckpt["model"])

    model_script = torch.jit.script(model)
    output_file = f"{args.ckpt.split('/')[-1].split('.')[0]}.jit"
    model_script.save(os.path.join(args.output_dir, output_file))


if __name__ == "__main__":
    main()