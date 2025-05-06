#!/bin/bash 
#SBATCH --job-name=embed_ant
#SBATCH --array=0-29%8  # Total tasks % the max number you want to run simultaneously 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=23:59:59
#SBATCH --output=mrq_embeds/%x_%A_%a.out
#SBATCH --error=mrq_embeds/%x_%A_%a.err

module purge
module load anaconda3/2021.11
conda init bash
source ~/.bashrc

conda activate mrq_gpu 

export WANDB_API_KEY=028b65afa7cbbb082902c650f03bfd3801495a70
export WANDB_MODE=offline

ZS_ZSA_ZA_LIST=(
  "512|512|256"
  "256|256|128"
  "128|128|64"
  "64|64|32"
  "32|32|16"
  "16|16|8"
)

SEEDS=(0 1 2 3 4)

COMBOS=() 

for combo in "${ZS_ZSA_ZA_LIST[@]}"; do
  IFS='|' read -r ZS ZSA ZA <<< "$combo"
  for SEED in "${SEEDS[@]}"; do
    COMBOS+=("${ZS}|${ZSA}|${ZA}|${SEED}")
  done
done

IDX=$SLURM_ARRAY_TASK_ID

#-----sanity checks-------
echo "Number of combos: ${#COMBOS[@]}" 
echo "IDX: $IDX" 
echo "combo: ${COMBOS[$IDX]}"

IFS='|' read -r ZS ZSA ZA SEED  <<< "${COMBOS[$IDX]}"

echo "ZS: $ZS"
echo "ZSA: $ZSA"
echo "ZA: $ZA"
echo "SEED: $SEED"


python new_main.py \
    --env Gym-Ant-v4 \
    --seed "${SEED}" \
    --total_timesteps 1000000 \
    --device cpu \
    --zs-dim "${ZS}" \
    --zsa-dim "${ZSA}" \
    --za-dim "${ZA}"
 
