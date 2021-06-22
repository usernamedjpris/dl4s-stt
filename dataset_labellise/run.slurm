#!/bin/bash
#SBATCH --job-name=merge            # nom du job
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=1         # nombre de taches MPI par noeud
#SBATCH --time=10:00:00             # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=merge%j.out          # nom du fichier de sortie
#SBATCH --error=merge%j.out           # nom du fichier d'erreur (ici en commun avec la sortie)
#SBATCH -A sbg@cpu

# on se place dans le repertoire de soumission
#cd ${SLURM_SUBMIT_DIR}
 
# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
conda deactivate
module load ffmpeg
module load anaconda-py3/2020.11
 
# echo des commandes lancees
set -x
 
# execution
conda activate dl4s-stt
srun python merge.py --wp1 ${ALL_CCFRSCRATCH}/WP1/ --cv ${ALL_CCFRSCRATCH}/commonvoice/ 