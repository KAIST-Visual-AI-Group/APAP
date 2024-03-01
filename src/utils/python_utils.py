"""
python_utils.py

A collection of utility functions based on Python standard library.
"""

import os
import shlex
import subprocess
from typing import List, Iterable

from typeguard import typechecked


@typechecked
def split(iterable: Iterable, num_chunk: int) -> Iterable:
    """Splits an iterable into a number of chunks"""
    k, m = divmod(len(iterable), num_chunk)
    chunks = [iterable[i * k + min(i, m):(i+1) * k + min(i+1, m)] for i in range(num_chunk)]
    return chunks

def launch_job_on_cpu(cmds: List[str]) -> None:
    """
    Launches a job on a GPU.
    """
    for cmd in cmds:
        p = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output, error = p.communicate()
 
        print(output.decode("utf-8"))
        print(error.decode("utf-8"))

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
