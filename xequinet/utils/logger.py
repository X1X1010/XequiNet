import logging
import os


class ZeroLogger:
    """
    Only logging in rank Zero process.
    """

    def __init__(
        self,
        is_rank0: bool = False,
        output_dir: str = "./",
        log_file: str = "loss.log",
        resume: bool = False,
        stream_name: str = "train",
    ) -> None:
        """
        Args:
            `is_rank0`: Whether the process is rank 0.
            `output_dir`: Output directory for log file.
            `log_file`: Name of the file logger.
            `resume`: Whether to resume logging to the same file.
            `stream_name`: Name of the stream logger.
        """
        self.output_dir = output_dir
        self.log_file = log_file
        self.resume = resume
        self.stream_name = stream_name
        if is_rank0:
            self.f = self.get_file_logger()  # file logger
            self.s = self.get_stream_logger()  # stream logger
        else:
            self.f = NoOp()
            self.s = NoOp()

    def get_file_logger(self) -> logging.Logger:
        file_logger = logging.getLogger(self.log_file.split(".")[0])
        file_logger.setLevel(logging.DEBUG)
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt=datefmt,
        )
        log_file = os.path.join(self.output_dir, self.log_file)
        if not self.resume:
            with open(log_file, "w") as f:
                pass
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_logger.addHandler(file_handler)
        return file_logger

    def get_stream_logger(self) -> logging.Logger:
        stream_logger = logging.getLogger(self.stream_name)
        stream_logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_logger.addHandler(stream_handler)
        return stream_logger


# no_op method/object that accept every signature
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            pass

        return no_op
