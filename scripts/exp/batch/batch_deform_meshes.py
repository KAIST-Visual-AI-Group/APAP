"""
batch_deform_meshes.py

A script for batch mesh deformation experiments.
"""

from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import shlex
from shutil import copyfile
import subprocess
from typing import List

import json
from jaxtyping import jaxtyped
from typeguard import typechecked
import tyro

from src.utils.python_utils import (
    split,
    launch_job_on_gpu,
)

# global variables
SRC_SCRIPT_PATH = Path(__file__).parent / "../deform_meshes.py"
"""The path to the script to execute"""
assert SRC_SCRIPT_PATH.exists(), f"Script path {SRC_SCRIPT_PATH} does not exist"


@dataclass
class Args:
    
    data_list_path: Path
    """Path to the file holding the list of data"""
    out_root: Path
    """Root directory of the outputs"""
    wandb_grp: str = "deform_meshes"
    """WandB group name"""
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    """GPU IDs to use"""


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:
    
    assert args.data_list_path.suffix == ".txt", (
        f"Expected .txt file. Got {str(args.data_list_path.suffix)}"
    )
    
    # create output root directory
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {str(args.out_root)}")

    # identify input files
    data_list = []
    with open(args.data_list_path, "r") as f:
        lines = f.readlines()
        for l in lines:
            if l.startswith("#"):  # skip commented lines
                continue
            mesh_file, handle_file, anchor_file, lora_dir = l.split(",")
            mesh_file = Path(mesh_file.strip())
            handle_file = Path(handle_file.strip())
            anchor_file = Path(anchor_file.strip())
            assert mesh_file.exists(), f"Mesh file {mesh_file} does not exist"
            assert handle_file.exists(), f"Handle file {handle_file} does not exist"
            assert anchor_file.exists(), f"Anchor file {anchor_file} does not exist"
            if lora_dir.strip().lower() != "none":
                lora_dir = Path(lora_dir.strip())
                assert lora_dir.exists(), f"LoRA checkpoint {lora_dir} does not exist"
            else:
                lora_dir = None

            data = {
                "mesh_file": mesh_file,
                "handle_file": handle_file,
                "anchor_file": anchor_file,
                "lora_dir": lora_dir,
            }
            data_list.append(data)

    # copy script file
    dst_file = args.out_root / "test_script.py"
    copyfile(SRC_SCRIPT_PATH, dst_file)
    assert dst_file.exists(), f"Copied file {dst_file} does not exist"

    # create a process pool
    gpu_ids = args.gpu_ids
    num_gpu = len(gpu_ids)
    print(f"[!] Available GPUs: {num_gpu}")
    pool = ThreadPool(num_gpu)

    # split data into chunks
    data_chunks = split(data_list, num_gpu)
    num_data = 0
    for chunk in data_chunks:
        num_data += len(chunk)
    assert len(data_list) == num_data, (
        f"Split is incorrect. {len(data_list)} != {num_data} (Expected)"
    )

    # launch job(s)
    for gpu_id, data_chunk in zip(gpu_ids, data_chunks):

        # populate commands
        cmds = []
        for data in data_chunk:
            mesh_file = data["mesh_file"]
            handle_file = data["handle_file"]
            anchor_file = data["anchor_file"]
            lora_dir = data["lora_dir"]

            # lookup mesh name
            with open(mesh_file.parent / "metadata.json", "r") as f:
                mesh_metadata = json.load(f)
                mesh_name = mesh_metadata["object_name"]

            # generate output path
            handle_id, anchor_id = handle_file.parent.stem, anchor_file.parent.stem
            out_dir = args.out_root / f"mesh-{mesh_name}_handle-{str(handle_id)}_anchor-{str(anchor_id)}"

            cmd = (
                f"""python {str(dst_file)}
                --mesh-file {str(mesh_file)}
                --handle-file {str(handle_file)}
                --anchor-file {str(anchor_file)}
                --out-dir {str(out_dir)}
                --wandb-grp {args.wandb_grp}
                --lora-dir {str(lora_dir)}
                """
            )

            cmds.append(cmd)

        # push the commands to the pool
        pool.apply_async(launch_job_on_gpu, (cmds, gpu_id, ))
    
    try:
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        
        print("Keyboard interrupt received. Terminating...")
        exit(0)

    print("[!] Done")


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )