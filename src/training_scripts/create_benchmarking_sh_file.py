import os

tasks = ["ANOMALY", "MULTI_CLASS_CLS", "MULTI_LABEL_CLS", "H_LABEL_CLS"]
versions = ["2.4.2", "2.5.0"]
gpu_count = 4  # total available GPUs

output_dir = "/home/rgangire/workspace/jobs/NonSlurmBM"
os.makedirs(output_dir, exist_ok=True)

template = """#!/bin/bash

source ~/miniforge3/etc/profile.d/conda.sh
conda activate otx{version_nodot}

conda info

export CUDA_VISIBLE_DEVICES={gpu_index}
export PYTHONPATH="${{PYTHONPATH}}:/home/rgangire/workspace/code/OTX/{version_nodot}/training_extensions/src"

export TASK={task}
export DATA_ROOT=/home/rgangire/workspace/datasets/
export NUM_REPEAT=5
export EVAL_UPTO=train
export DEVICE="cuda"
export OUTPUT_ROOT=/home/rgangire/workspace/Results/BenchMarking/{task}/OTX_{version}/

cd /home/rgangire/workspace/code/OTX/{version_nodot}/training_extensions/


python -u tests/perf_v2/run.py --task ${{TASK}} \\
    --data-root ${{DATA_ROOT}} \\
    --num-repeat ${{NUM_REPEAT}} \\
    --eval-upto ${{EVAL_UPTO}} \\
    --device ${{DEVICE}} \\
    --output-root ${{OUTPUT_ROOT}}
"""

counter = 0
for task in tasks:
    task_lc = task.lower()
    for version in versions:
        version_nodot = version.replace(".", "")
        gpu_index = counter % gpu_count
        filename = f"{task_lc}_{version_nodot}_gpu{gpu_index}.sh"
        filepath = os.path.join(output_dir, filename)

        script_content = template.format(
            task=task,
            version=version,
            version_nodot=version_nodot,
            gpu_index=gpu_index
        )

        with open(filepath, "w") as f:
            f.write(script_content)
        os.chmod(filepath, 0o755)  # make it executable

        counter += 1
        print(f"Created script: {filename}")


print(f"✅ Generated {counter} scripts in '{output_dir}'")

