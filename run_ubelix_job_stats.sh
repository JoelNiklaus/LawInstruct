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

cd /storage/workspaces/inf_fdn/hpc_nfp77/joel/LawInstruct/statistics
conda activate lawinstruct
export HF_DATASETS_CACHE="/storage/workspaces/inf_fdn/hpc_nfp77/joel/.cache"
module load git-lfs/2.4.2

python compute_agg_stats.py
python analyze_agg_stats.py


# IMPORTANT:
# Run with                  sbatch run_ubelix_job.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=epyc2 --qos=job_epyc2 --mem=64G --cpus-per-task=16 --time=02:00:00 --pty /bin/bash
