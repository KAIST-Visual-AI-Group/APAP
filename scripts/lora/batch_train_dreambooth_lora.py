"""
A script for launching paralle processes for training LoRA.
"""

from dataclasses import dataclass, field
import json
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import shlex
import subprocess
from typing import List, Literal, Optional

from typeguard import typechecked
import tyro

from src.utils.python_utils import split


@dataclass
class Args:

    data_list_path: Path
    """The path to the list of data"""
    exp_group_name: str
    """The name of the current experiment group"""
    out_root: Path
    """The root directory of the output weights"""
    # data_list: List[str] = field(default_factory=lambda: DATA_LIST)
    # """The list of data"""
    project_name: str = "apap"
    """The name of the project"""
    model_type: Literal[
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1-base",
    ] = "stabilityai/stable-diffusion-2-1-base"
    """The type of the model to fine-tune"""
    instance_prompt: Optional[str] = None
    """The prompt for the instance"""
    special_token: str = "sks"
    """The special token for the instance"""
    train_batch_size: int = 2
    """The batch size for training"""
    learning_rate: float = 5e-4
    """The learning rate"""
    max_train_steps: int = 200
    """The maximum number of training steps"""
    train_epochs: int = 200
    """The maximum number of training epochs"""
    validation_epochs: int = 50
    """The number of epochs between validations"""
    rank: int = 16
    """The rank of LoRA matrices"""
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    """The IDs of GPUs to use"""
    with_prior_preservation: bool = False
    """Whether to use prior preservation loss"""
    num_class_images: int = 200
    """The number of class images to use for prior preservation loss"""

@typechecked
def generate_command_dreambooth(
    script_path: Path,
    pretrained_model_name: Literal[
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1-base",
    ],
    instance_data_dir: Path,
    output_dir: Path,
    project_name: str,
    instance_prompt: str,
    validation_prompt: str,
    learning_rate: float,
    train_batch_size: int,
    max_train_steps: int = 800,
    train_epochs: int = 100,
    validation_epochs: int = 50,
    rank: int = 4,
    with_prior_preservation: bool=False,
    num_class_images: int=5,
    class_prompt: str=None,
) -> str:
    """Generates a command for launching a training job"""
    command = f"python {str(script_path.resolve())} \
        --pretrained_model_name_or_path='{str(pretrained_model_name)}'  \
        --instance_data_dir='{str(instance_data_dir.resolve())}' \
        --output_dir='{str(output_dir.resolve())}' \
        --instance_prompt='{str(instance_prompt)}' \
        --resolution=512 \
        --train_batch_size={str(train_batch_size)} \
        --gradient_accumulation_steps=1 \
        --checkpointing_steps=100 \
        --learning_rate={str(learning_rate)} \
        --project_name='{str(project_name)}' \
        --report_to='wandb' \
        --lr_scheduler='constant' \
        --lr_warmup_steps=0 \
        --mixed_precision=fp16 \
        --max_train_steps={max_train_steps} \
        --num_train_epochs={train_epochs} \
        --validation_prompt='{str(validation_prompt)}' \
        --validation_epochs={str(validation_epochs)} \
        --rank={rank} \
        --seed='0'"
    if with_prior_preservation:
        class_data_dir = output_dir / "class_dir"
        class_data_dir.mkdir(parents=True, exist_ok=True)
        command += f" --with_prior_preservation \
            --class_data_dir='{str(class_data_dir.resolve())}' \
            --class_prompt='{str(class_prompt)}' \
            --num_class_images={num_class_images}"

    return command

@typechecked
def run_job(commands: List[str], exp_group_name: str, gpu_id: int):
    """Runs a list of jobs on a designated GPU"""
    for command in commands:
        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(
                os.environ,
                WANDB_RUN_GROUP=str(exp_group_name),
                CUDA_VISIBLE_DEVICES=str(gpu_id),
            ),
        )
        output, error = process.communicate()

        # TODO: Save the log in files
        print(output.decode("utf-8"))
        print(error.decode("utf-8"))

@typechecked
def main(args: Args) -> None:
    
    # parse arguments
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    model_type = args.model_type
    train_batch_size = args.train_batch_size
    learning_rate = args.learning_rate
    max_train_steps = args.max_train_steps
    train_epochs = args.train_epochs
    rank = args.rank
    with_prior_preservation = args.with_prior_preservation
    num_class_images = args.num_class_images

    # identify script file to execute
    script_file = Path(__file__).parents[0] / "train_dreambooth_lora.py"
    assert script_file.exists(), f"Script file {script_file} does not exist"

    # create a process pool
    gpu_ids = args.gpu_ids
    num_gpu = len(gpu_ids)
    print(f"[!] Available GPUs: {num_gpu}")
    pool = ThreadPool(num_gpu)

    # launch chunks of jobs simultaneously
    if False:
        data_list = args.data_list
    else:
        data_list = []
        with open(args.data_list_path, mode="r") as f:
            lines = f.readlines()
            for l in lines:
                if not l.startswith("#"):
                    data_list.append(l.strip())

    # split data into multiple chunks
    data_chunks = split(data_list, num_gpu)
    num_data = 0
    for data_chunk in data_chunks:
        num_data += len(data_chunk)
    assert num_data == len(data_list), (
        f"Data is not split correctly: {num_data} vs {len(data_list)}"
    )

    for gpu_index, data_chunk in zip(gpu_ids, data_chunks):
        commands = []
        # for data_dir in data_chunk:
        for item in data_chunk:

            # parse item from the config file
            object_name, data_dir = item.split(",")
            data_dir = Path(data_dir.strip())
            assert data_dir.exists(), f"Data directory {data_dir} does not exist"
            assert data_dir.is_dir(), f"Data directory {data_dir} is not a directory"

            # generate command
            instance_prompt = f"a photo of {args.special_token} {object_name}"

            class_prompt = None
            if args.instance_prompt is not None:
                instance_prompt = args.instance_prompt
            if args.with_prior_preservation:
                class_prompt = f"a photo of {object_name}"
            validation_prompt = instance_prompt
            print(f"Instance prompt: {str(instance_prompt)}")

            sample_name = data_dir.parents[1].stem
            output_dir = out_root / sample_name / data_dir.stem
            instance_data_dir = data_dir  # TODO: drop this redundant variable
            assert instance_data_dir.exists(), (
                f"Instance data directory {instance_data_dir} does not exist"
            )

            # save metadata
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = output_dir / "metadata.json"
            with open(metadata_file, mode="w") as f:
                metadata = {
                    "prompt": instance_prompt,
                    "special_token": args.special_token,
                    "object_name": object_name,
                    "data_dir": str(data_dir.resolve()),
                }
                f.write(json.dumps(metadata, indent=4))
            print(f"Output directory: {str(output_dir.resolve())}")

            command = generate_command_dreambooth(
                script_path=script_file,
                pretrained_model_name=model_type,
                instance_data_dir=instance_data_dir,
                output_dir=output_dir,
                project_name=args.project_name,
                instance_prompt=instance_prompt,
                validation_prompt=validation_prompt,
                train_batch_size=train_batch_size,
                learning_rate=learning_rate,
                max_train_steps=max_train_steps,
                train_epochs=train_epochs,
                rank=rank,
                with_prior_preservation=with_prior_preservation,
                num_class_images=num_class_images,
                class_prompt=class_prompt,
            )

            commands.append(command)

        # push the command to the pool
        pool.apply_async(run_job, (commands, args.exp_group_name, gpu_index, ))

    pool.close()
    pool.join()

    print("[!] Done")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
