#!/bin/bash

# 1. Initialiser Conda (adapte le chemin vers ton anaconda/miniconda)
source /home/liteul/anaconda3/etc/profile.d/conda.sh

# 2. Activer ton environnement
conda activate nmma_env

# 3. Lancer le script python avec l'argument donné par Condor ($1)
python /home/liteul/memoir_code/run_analysis.py $1