"""
utils.py

Utility functions for launching jobs on GPUs.
"""

import os
import shlex
import subprocess
from typing import List



def launch_job_on_gpu(cmds: List[str], gpu_id: int) -> None:
    """
    Launches a job on a GPU.
    """
    for cmd in cmds:
        p = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu_id),
            ),
        )
        output, error = p.communicate()
 
        print(output.decode("utf-8"))
        print(error.decode("utf-8"))
