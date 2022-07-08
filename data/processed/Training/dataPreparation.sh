#!/bin/bash
wget https://sourceforge.net/projects/deeprotgo/files/data/processed/Training/LevSim_BP_Euka.h5 --quiet
wget https://sourceforge.net/projects/deeprotgo/files/examples/train_NK_Euka_CC/DeeProtGO_PSD_Emb_Taxon_Euka_CC_NK.pt --quiet
mkdir DeeProtGO/examples/train_NK_EUKA_CC
mv *.pt DeeProtGO/examples/train_NK_EUKA_CC/

tar -xzvf DeeProtGO/data/processed/Training/Emb_BP_Euka.h5.tar.gz 
tar -xzvf DeeProtGO/data/processed/Training/netOut_BP_Euka.h5.tar.gz 
tar -xzvf DeeProtGO/data/processed/Training/Emb_CC_Euka.h5.tar.gz
tar -xzvf DeeProtGO/data/processed/Training/netOut_CC_Euka.h5.tar.gz

mv *.h5 DeeProtGO/data/processed/Training/
