#!/bin/bash
#SBATCH --job-name=Seq              # nom du job
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=1         # nombre de taches MPI par noeud
#SBATCH --time=00:01:00             # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=Seq%j.out          # nom du fichier de sortie
#SBATCH --error=Seq%j.out           # nom du fichier d'erreur (ici en commun avec la sortie)
 
# on se place dans le repertoire de soumission
cd ${SLURM_SUBMIT_DIR}
 
# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
module load ffmpeg
 
# echo des commandes lancees
set -x
 
# execution
python merge --wp1 ${ALL_CCFRSCRATCH}/WP1/ --cv ${ALL_CCFRSCRATCH}/commonvoice/ 