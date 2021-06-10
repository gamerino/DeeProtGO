#!/bin/bash
wget https://sourceforge.net/projects/deeprotgo/files/data/processed/Training/LevSim_BP_Euka.h5 --quiet
mv LevSim_BP_Euka.h5 DeeProtGO/data/processed/Training/
tar -xzvf DeeProtGO/data/processed/Training/Emb_BP_Euka.h5.tar.gz 
tar -xzvf DeeProtGO/data/processed/Training/netOut_BP_Euka.h5.tar.gz 

