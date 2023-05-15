#!/bin/bash
#SBATCH --job-name="LawInstruct"
#SBATCH --mail-user=joel.niklaus@inf.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --qos=job_epyc2
#SBATCH --partition=epyc2

# Put your code below this line

cd /storage/workspaces/inf_fdn/hpc_nfp77/joel/LawInstruct
conda activate lawinstruct
export HF_DATASETS_CACHE="/storage/workspaces/inf_fdn/hpc_nfp77/joel/.cache"


# Define the language modes and instruction bank sizes
language_modes=("english" "multilingual")
instruction_bank_sizes=(10) # (1 2 5 10)

# Loop over the language modes and instruction bank sizes
for language_mode in "${language_modes[@]}"
do
    for instruction_bank_size in "${instruction_bank_sizes[@]}"
    do
        # Invoke the Python script with the desired arguments
        python build_instruction_datasets.py --language_mode "$language_mode" --instruction_bank_size "$instruction_bank_size" --datasets legal --build_from_scratch
    done
done


# xz --list data/*.xz
# python build_num_shards_dict.py

# IMPORTANT:
# Run with                  sbatch run_ubelix_job.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=epyc2 --qos=job_epyc2 --mem=64G --cpus-per-task=16 --time=02:00:00 --pty /bin/bash
