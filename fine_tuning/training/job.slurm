#!/bin/bash
#SBATCH --job-name=pretraining            # nom du job
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=1         # nombre de taches MPI par noeud
#SBATCH --ntasks=8         # nombre de taches MPI par noeud

#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread         # nombre de taches MPI par noeud

#SBATCH --time=10:00:00             # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=pretraining.out          # nom du fichier de sortie
#SBATCH --error=pretraining.err           # nom du fichier d'erreur (ici en commun avec la sortie)
#SBATCH -A sbg@gpu



MODULE_ENV="anaconda-py3"
RUN_DIR="$ALL_CCFRWORK/STT/dl4s-stt/fine_tuning"
FAIRSEQ="$ALL_CCFRWORK/libs/fairseq/examples/wav2vec/config/pretraining"
DATA="$ALL_CCFRWORK/STT/stt_logs"
CONFIG="wav2vec2_base_librispeech"
 
# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
conda deactivate
module load $MODULE_ENV
conda activate dl4s-stt
 
# execution
cd $RUN_DIR
fairseq-hydra-train task.data=$DATA/pretraining_10 distributed_training.distributed_world_size=0 --config-dir $FAIRSEQ --config-name $CONFIG