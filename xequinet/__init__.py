import os
import warnings

# check if the process is the master process
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
else:
    local_rank = 0

# mute warnings for non-master processes
if local_rank != 0:
    warnings.filterwarnings("ignore")
