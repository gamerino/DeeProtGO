# ==============================================================================
# IBB-CONICET-UNER
# sinc(i)-CONICET-UNL. http://sinc.unl.edu.ar/
# G. Merino, et al.
# gmerino@ingenieria.uner.edu.ar
# gmerino@sinc.unl.edu.ar
# ==============================================================================

import pandas as pd
import numpy as np
import torch
import os
import random
from sklearn.utils import shuffle
from torch.utils import data
        
class Dataloader(data.Dataset): 
    """Dataloader for training and testing DeeProtGO."""
    def __init__(self, dirData, posEntriesFile, negEntriesFile, netOutFile,
                 inputData1, inputData2 = None, inputData3 = None, inputData4 = None, inputData5 = None,
                 inputData6 = None, randomize = True, samplingNegPerc = 1, idxFiltGOFile = None):
        """
        Args:
            dirData (str): Path to the common-directory where data files are deposited.
            posEntriesFile (str): Name of the file with the names of positives proteins. 
            negEntriesFile (str): Name of the file with the names of negatives proteins.
            netOutFile (str): Filename where the desired network output is. Proteins in rows, GO terms in columns.
            inputDataN (str): Filename where the N-th input data is. Proteins in rows, features in columns. N = 1, 2, 
                              3, 4, 5, 6.
            randomize (bool): Logical indicating if randomization should be done.
                               Default = True
            samplingNegPerc (float): Floating number between 0 and 1 indicating the proportion of negative proteins 
                                     that should be kept.
                                     Default = 1 (All negative proteins)
            idxFiltGOFile (str): Name of the file with indexes of GO terms of the netOut that should be kept (1-based indexes)
                                 Default = None (All GO terms are kept)
        """
        print("Loading data...")
        #self.device=device     
        # data loading        
        posEntryNames = np.loadtxt(dirData + posEntriesFile, delimiter = '\t', dtype = 'str').tolist()
        negEntryNames = np.loadtxt(dirData + negEntriesFile, delimiter = '\t', dtype = 'str').tolist()
        if samplingNegPerc < 1:
            negIdx = random.sample(range(len(negEntryNames)), round(samplingNegPerc * len(negEntryNames)))
        else:
            negIdx = range(len(negEntryNames))
        fullEntryNames = np.concatenate([posEntryNames, [negEntryNames[i] for i in negIdx]], axis = 0)
        if randomize:
            fullEntryNames = shuffle(fullEntryNames)
        self.cases = dict()  
        self.cases["labels"] = fullEntryNames
        netOut = pd.read_hdf(dirData + netOutFile, "df").loc[ fullEntryNames, :]
        if isinstance(idxFiltGOFile, str):
            idxFiltGO = np.loadtxt(dirData + idxFiltGOFile, delimiter = "\t", dtype = 'int')  
            idxFiltGO = idxFiltGO - 1  # because are 1-based indexes
            netOut = netOut.iloc[:, idxFiltGO]
        inData1 = pd.read_hdf(dirData + inputData1, "df").loc[fullEntryNames,:]
        if isinstance(inputData2, str):
            inData2 = pd.read_hdf(dirData + inputData2, "df").loc[fullEntryNames, :]
        else:
            inData2 = None
        if isinstance(inputData3, str):
            inData3 = pd.read_hdf(dirData + inputData3, "df").loc[fullEntryNames, :]
        else:
            inData3 = None
        if isinstance(inputData4, str):
            inData4 = pd.read_hdf(dirData + inputData4, "df").loc[fullEntryNames, :]
        else:
            inData4 = None
        if isinstance(inputData5, str):
            inData5 = pd.read_hdf(dirData + inputData5, "df").loc[fullEntryNames, :]
        else:
            inData5 = None
        if isinstance(inputData6, str):
            inData6 = pd.read_hdf(dirData + inputData6, "df").loc[fullEntryNames, :]
        else:
            inData6 = None            
        self.cases["GOLabels"] = netOut.columns
        self.cases["negLabels"] = [negEntryNames[i] for i in negIdx]
        self.cases["netOut"] = netOut
        self.cases["inData1"] = inData1
        self.cases["inData2"] = inData2
        self.cases["inData3"] = inData3
        self.cases["inData4"] = inData4
        self.cases["inData5"] = inData5
        self.cases["inData6"] = inData6
        print("Dataset ready with %d cases ." %(len(self.cases[ "labels" ])))
    def get_output(self, proteins = np.array([])):
        if np.size(proteins) > 0:
            outData = self.cases["netOut"].loc[proteins,:]
        else:
            outData = self.cases["netOut"]
        return outData
    def get_labels(self, ind = np.array([])):
        if np.size(ind) > 0:
            return self.cases["labels"][ind]
        return self.cases["labels"]
    def get_inData1(self, proteins = np.array([])):
        if np.size(proteins) > 0:
            inData1 = self.cases["inData1"].loc[proteins, :]
        else:
            inData1 = self.cases["inData1"]
        return inData1
    def get_inData2(self, proteins = np.array([])):
        if np.size(proteins) > 0:
            inData2 = self.cases["inData2"].loc[proteins, :]
        else:
            inData2 = self.cases["inData2"]
        return inData2
    def get_inData3(self, proteins = np.array([])):
        if np.size(proteins) > 0:
            inData3 = self.cases["inData3"].loc[proteins, :]
        else:
            inData3 = self.cases["inData3"]
        return inData3
    def get_inData4(self, proteins = np.array([])):
        if np.size(proteins) > 0:
            inData4 = self.cases["inData4"].loc[proteins, :]
        else:
            inData4 = self.cases["inData4"]
        return inData4
    def get_inData5(self, proteins = np.array([])):
        if np.size(proteins) > 0:
            inData5 = self.cases["inData5"].loc[proteins, :]
        else:
            inData5 = self.cases["inData5"]
        return inData5
    def get_inData6(self, proteins = np.array([])):
        if np.size(proteins) > 0:
            inData6 = self.cases["inData6"].loc[proteins, :]
        else:
            inData6 = self.cases["inData6"]
        return inData6
    def get_GOTerm(self, ind = np.array([])):
        if np.size(ind) > 0:
            return self.cases["GOLabels"][ind]
        else:
            return self.cases["GOLabels"]
    def get_batch(self, proteins):
    #"""Returns input data tensor and output ready for training using the cases contained in "proteins"."""
        inData1Torch = torch.from_numpy(self.get_inData1(proteins).values)
        if self.cases["inData2"] is not None:
            inData2Torch = torch.from_numpy(self.get_inData2(proteins).values)
        else:
            inData2Torch = None
        if self.cases["inData3"] is not None:
            inData3Torch = torch.from_numpy(self.get_inData3(proteins).values)
        else:
            inData3Torch = None
        if self.cases["inData4"] is not None:
            inData4Torch = torch.from_numpy(self.get_inData4(proteins).values)
        else:
            inData4Torch = None
        if self.cases["inData5"] is not None:
            inData5Torch = torch.from_numpy(self.get_inData5(proteins).values)
        else:
            inData5Torch = None
        if self.cases["inData6"] is not None:
            inData6Torch = torch.from_numpy(self.get_inData6(proteins).values)
        else:
            inData6Torch = None
        outDataTorch = torch.from_numpy(self.get_output(proteins).values)
        return inData1Torch, inData2Torch, inData3Torch, inData4Torch, inData5Torch, inData6Torch, outDataTorch
    
