# Predicting protein functions using DeeProtGO

This repository contains the full source code and a detailed example of a case-of-use of `DeeProtGO`, a deep learning model for automatic function prediction by integrating protein knowledge from multiple databases.

The following sections contain the steps and the libraries used in each of them. All datasets and libraries are open sourced and free to use. If you use any of the following methods, please cite them properly:

**G.A. Merino, R. Saidi, D.H. Milone, G. Stegmayer, M. Martin. *Hierarchical deep learning for predicting GO annotations by integrating heterogeneous protein knowledge*, 2021.**

## 1. Introduction

This workflow describes the steps required to predict [GO](http://geneontology.org/) terms for *No-knowledge* (NK) proteins, that is, without experimental annotations in any of the GO sub-ontologies at a reference time, but that have accumulated at least one GO term with an experimental evidence code during a growth period. 

Data used for developing and evaluating the proposed models were obtained from different protein-knowledge databases, considering the proteins provided in the [CAFA3 challenge](https://www.biofunctionprediction.org/cafa/). The CAFA3 dates were considered:
-  *t*<sub>-1</sub> is when the challenge was released (09/2016), in which the sets of training and target proteins were provided to the participants; 
-  *t*<sub>0</sub>, the deadline for participants submissions of the predictions for the target proteins (02/2017); and 
-  *t*<sub>1</sub> is when benchmark proteins were collected for assessment (11/2017). 
Thus, the CAFA3 benchmark dataset is composed of those target proteins that have, at least, one new functional annotation added during the growth period between  *t*<sub>0</sub> and  *t*<sub>1</sub>.

Since DeeProtGO was developed to learn new annotations gained during a time gap, it is trained here with proteins that gained GO terms during the growth period defined between *t*<sub>-1</sub>  and  *t*<sub>0</sub>, and evaluated on NK proteins of the CAFA3 benchmark dataset (at  *t*<sub>1</sub>).


## 2. Data preparation

**Note**: All the results generated in this section are provided in data/intermediate and data/processed directories.

The [data preparation notebook](https://colab.research.google.com/drive/1uZeYf6geBsn9OqJy5hwv2VSj-Rix2k2z?authuser=1) is provided with instructions to build all input and output data required for training DeeProtGO for predicting BP terms for NK proteins and for testing it with CAFA3 benchmark proteins. Doing so may take several hours. 


## 3. Model training

**Note**: All the results generated in this section are provided in the examples/train_NK_EUKA_BP directory.

The [DeeProtGO training notebook](https://colab.research.google.com/drive/1UwjkXnMB3Tte-8xR7AtgGjCAD_BIhiOQ?authuser=1) is provided with all the steps required to train DeeProtGO for predicting BP terms for NK proteins.


## 4. Model testing on CAFA3 benchmark

The [DeeProtGO testing notebook](https://colab.research.google.com/drive/1tKTGvc1MKFNe2bawfTm9aQdmx_a6iMAv?authuser=1) is provided with all the steps required to evaluate DeeProtGO when predicting BP terms for NK proteins of eukarya organisms in the CAFA3 benchmark dataset.


## 5. Repository organization

The following scheme illustrates the organization of this repository and contains a detailed description of each folder and file.


```
Protein function prediction
│
├── data             -> All data for preparing files for DeeProtGO training and testing.
│   │
│   ├── extras       -> Files useful for datasets preparation.
│   │   │
│   │   └── gene_ontology_edit.obo        -> GO file with the ontology used for CAFA3.
│   │ 
│   ├── intermediate -> Intermediate files generated during training and benchmark datasets preparation. 
│   │   │
│   │   ├── propAnnot_Bench_Euka_BP.tab           -> Propagated GO annotations gained between  t0 and  t1 for benchmark 
│   │   │                                            proteins.   
│   │   └── propAnnot_Train_Euka_BP.tab           -> Propagated GO annotations gained between  t-1 and  t0 for training 
│   │                                                proteins. 
│   ├── processed    -> Datasets used for training and testing DeeProtGO when predicting annotations for NK proteins. 
│   │   │
│   │   ├── Benchmark  -> Data for NK proteins from eukarya organisms in CAFA3 benchmark for BP.
│   │   │       │
│   │   │       ├── Emb_BP_Euka.h5          -> SeqVec embeddings.
│   │   │       │
│   │   │       ├── LevSim_BP_Euka.h5       -> Proteins similarity based on edit distance.
│   │   │       │
│   │   │       ├── NegEntries_Euka_BP.tab  -> UniProt entry names of negative proteins.
│   │   │       │
│   │   │       ├── netOut_BP_Euka.h5       -> One-hot encoding matrix representing GO BP terms of benchmark proteins.
│   │   │       │
│   │   │       ├── PosEntries_Euka_BP.tab  -> UniProt entry names of positive proteins.
│   │   │       │
│   │   │       └── Taxon_BP_Euka.h5        -> One-hot encoding matrix representing proteins taxon.
│   │   │       
│   │   └── Training   -> Data used for training DeeProtGO to predict GO BP terms of  NK proteins from eukarya organisms.
│   │           │
│   │           ├── dataPreparation.sh                  -> Bash script for downloading proteins similarity file and 
│   │           │                                          uncompressing *tar.gz files
│   │           ├── Emb_BP_Euka.h5.tar.gz               -> SeqVec embeddings.
│   │           │
│   │           ├── GOTermsPropRel_Euka_BP_train.tab    -> One-hot encoding of relationships between GO terms, used for scores
│   │           │                                          propagation.
│   │           ├── NegEntries_Euka_BP.tab              -> UniProt entries of negative proteins.
│   │           │
│   │           ├── netOut_BP_Euka.h5.tar.gz            -> One-hot encoding matrix representing GO BP terms of training
│   │           │                                           proteins.
│   │           ├── PosEntries_Euka_BP.tab              -> UniProt entries of positive proteins.
│   │           │
│   │           └── Taxon_BP_Euka.h5                    -> One-hot encoding matrix representing proteins taxon.
│   │
│   └── raw  -> Metadata of training and benchmark proteins
│       │
│       ├── benchmarkNKEukaBPInfo.tab     -> Data of benchmark proteins
│       │
│       └── trainingNKEukaBPInfo.tab      -> Data of training proteins
|   					   
│
├── examples        -> Files obtained when running the DeeProtGO example provided here
│   │
│   └── train_NK_EUKA_BP       -> Files obtained during training of DeeProtGO for predicting BP terms of NK proteins
│       │                                           
│       ├── modelPreparation.sh                      -> Bash script for downlading a pre-trained DeeProtGO model.
│       │                                           
│       ├── test_13052021.log                        -> DeeProtGO performance on test partition of the training dataset.
│       │                                            
│       ├── train_13052021.log                       -> DeeProtGO loss on train partition of training the dataset.
│       │
│       └── valid_13052021.log                       -> DeeProtGO performance on validation partition of the training dataset.
|					     
│
├── scripts         -> Files obtained when running the DeeProtGO example provided here
│   │
│   ├── dataPreparation.ipynb      -> Python notebook for preparing the data for training and testing DeeProtGO
│   │
│   ├── DeeProtGOTesting.ipynb     -> Python notebook for training DeeProtGO
│   │
│   └── DeeProtGOTraining.ipynb    -> Python notebook for testing DeeProtGO
│
├── src             -> Source files of DeeProtGO
    │
    ├── dataloader.py      -> Class and methods for loading the data
    │
    ├── DNN.py             -> Deep neural network behind DeeProtGO
    │
    ├── DNNModel.py        -> DeeProtGO model and its functions
    │
    ├── earlyStop.py       -> Method for early stopping
    │
    ├── logger.py          -> Class and methods for saving performance measures and model loss during training and testing.
    │
    └── sampler.py         -> Class and methods for generating random batchs for being used during model training.
```






