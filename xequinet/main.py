import argparse
from xequinet.run import (
    run_train, run_test, run_infer, compile_model,
    run_opt, run_md, run_pimd,
)

def main() -> None:
    # parse config
    parser = argparse.ArgumentParser(description="XequiNet Entry Point")
    # task selection
    parser.add_argument(
        "task", type=str,
        choices=["train", "test", "infer", "opt", "jit", "md", "pimd"],
        help="Task selection.",
    )    
    # common arguments
    parser.add_argument(
        "--config", "-C", type=str, default="config.json",
        help="Configuration file of json format (default: config.json).",
    )
    parser.add_argument(
        "--warning", "-w", action="store_true",
        help="Whether to show warning messages",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=0, choices=[0, 1, 2],
        help="Verbose level (default: 0).",
    )
    # test
    parser.add_argument(
        "--ckpt", "-c", type=str, default=None,
        help="Checkpoint file for testing.",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32,
        help="Batch size for testing or inference. (default: 32)",
    )
    parser.add_argument(
        "--no-force", action="store_true",
        help="Whether testing without force when output mode is 'grad'.",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Whether testing force additionlly when output mode is 'scalar'.",
    )
    # inference
    parser.add_argument(
        "--delta", "-d", type=str, default=None,
        help="Base semi-empirical method of delta-learning model.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file name.",
    )
    # jit
    parser.add_argument(
        "--for-md", action="store_true",
        help="Whether the model is used for molecular dynamics.",
    )
    # geometry
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Maximum number of optimization steps.",
    )
    parser.add_argument(
        "--cons", type=str, default=None,
        help="Constraints file for optimization.",
    )
    parser.add_argument(
        "--freq", action="store_true",
        help="Calculate vibrational frequencies.",
    )
    parser.add_argument(
        "--numer", action="store_true",
        help="Calculate hessian with numerical second derivative.",
    )
    parser.add_argument(
        "--shm", action="store_true",
        help="Whether to write shermo input file.",
    )
    parser.add_argument(
        "--no-opt", action="store_true",
        help="Do not perform optimization.",
    )
    parser.add_argument(
        "--temp", "-T", type=float, default=298.15,
        help="Temperature for vibrational frequencies.",
    )
    # input
    parser.add_argument(
        "inp", type=str, default=None,
        help="""Input file.
            For inference and optimization, it should be xyz file.
            For molecular dynamics, it should be settings file.    
            """,
    )
    args = parser.parse_args()
    
    if not args.warning:
        import warnings
        warnings.filterwarnings("ignore")

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
    elif args.task == "pimd":
        run_pimd(args)
    else:
        raise NotImplementedError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()