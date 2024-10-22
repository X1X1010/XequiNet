import argparse

from xequinet.run import compile_model, run_infer, run_md, run_opt, run_test, run_train


def main() -> None:
    # parse config
    parser = argparse.ArgumentParser(description="XequiNet Entry Point")
    # task selection
    parser.add_argument(
        "task",
        type=str,
        choices=["train", "test", "infer", "opt", "jit", "md"],
        help="Task selection.",
    )
    # common arguments
    parser.add_argument(
        "--config",
        "-C",
        type=str,
        default="config.yaml",
        help="Configuration file of yaml format (default: config.yaml).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=0,
        help="Verbose level (default: 0).",
    )
    # test
    parser.add_argument(
        "--ckpt",
        "-c",
        type=str,
        default=None,
        help="Checkpoint file for testing.",
    )
    # inference
    parser.add_argument(
        "--forces",
        action="store_true",
        help="Whether to compute forces.",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Whether to compute stress.",
    )
    parser.add_argument(
        "--delta",
        "-d",
        type=str,
        default=None,
        help="Base semi-empirical method of delta-learning model.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file name.",
    )
    # jit compile
    parser.add_argument(
        "--fusion-strategy",
        type=str,
        default="DYNAMICS,3",
    )
    # geometry
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of optimization steps.",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        help="Constraints file for optimization.",
    )
    parser.add_argument(
        "--freq",
        action="store_true",
        help="Calculate vibrational frequencies.",
    )
    parser.add_argument(
        "--numerical",
        action="store_true",
        help="Calculate hessian with numerical second derivative.",
    )
    parser.add_argument(
        "--shermo",
        action="store_true",
        help="Whether to write shermo input file.",
    )
    parser.add_argument(
        "--no-opt",
        action="store_true",
        help="Do not perform optimization.",
    )
    parser.add_argument(
        "--temp",
        "-T",
        type=float,
        default=298.15,
        help="Temperature for vibrational frequencies.",
    )
    # stda
    parser.add_argument(
        "--xc",
        type=str,
        default="B3LYP",
        help="Exchange-correlation functional for SCF.",
    )
    parser.add_argument(
        "--nstates",
        type=int,
        default=5,
        help="Number of states for sTDA.",
    )
    parser.add_argument(
        "--as-init-guess",
        action="store_true",
        help="Use Fock matrix as initial guess for SCF, then run sTDA.",
    )
    # input
    parser.add_argument(
        "--input",
        "-in",
        type=str,
        default=None,
        help="""Input file.
            For inference and optimization, it should be xyz file.
            For molecular dynamics, it should be settings file.
            """,
    )
    args = parser.parse_args()

    if args.task == "train":
        run_train(args)
    elif args.task == "test":
        run_test(args)
    elif args.task == "infer":
        run_infer(args)
    elif args.task == "jit":
        compile_model(args)
    elif args.task == "opt":
        run_opt(args)
    elif args.task == "md":
        run_md(args)
    else:
        raise NotImplementedError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
