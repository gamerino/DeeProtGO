# ==============================================================================
# IBB-CONICET-UNER
# sinc(i)-CONICET-UNL. http://sinc.unl.edu.ar/
# G. Merino, et al.
# gmerino@ingenieria.uner.edu.ar
# gmerino@sinc.unl.edu.ar
# ==========================================================================

import pandas as pd
import numpy as np
import torch,io, pickle, os,random

class Sampler():
    """Partition sampler by batch."""
    def __init__(self, cases, fold, nfold, nbatch = 16, partsize = [.70, .10, .20]):
        """
        Args:
            cases (int): Number of proteins.
            fold (int): Number indicating the fold for which the partition sampling is required.
            nfold (int): Number of folds of the cross-validation procedure. 
            nbatch (int): Batch size used for model training
                         Default: 16
            partsize (list): Proportions of the cases that should be considered for training, validation and testing.
                            Default: [.70, .10, .20]                      
        """
        self.nbatch = nbatch
        nCases = len(cases)
        def rotate(l, n):
            return np.concatenate((l[-n:], l[:-n]))
        cases = rotate(cases, fold*nCases//nfold) # using different cases each fold, ensure that everyone is used at least once.
        nTrain = int(partsize[0]*nCases)
        nValid = int(partsize[1]*nCases)
        # avoiding errors for batches with only one sample        
        if(nTrain%nbatch == 1):
            nTrain-=1        
        self.train = cases[:nTrain]
        self.validation = cases[ nTrain:(nTrain + nValid) ]
        self.test = cases[ (nTrain+nValid): ]
    def batch_ind(self, part):
        proteins = getattr(self, part)
        if part != "test":
            L = len(proteins)
            random.shuffle(proteins)
            lowIdx = 0
            uppIdx = self.nbatch
            batchProt = []
            numBatches = int( L / self.nbatch )
            for l in range(numBatches):
                batchProt.append(proteins[lowIdx:uppIdx])
                lowIdx = uppIdx
                uppIdx += self.nbatch
            uppIdx -= self.nbatch
            if uppIdx < L:
                batchProt.append(proteins[uppIdx:])
        else:    
            batchProt = [proteins]
        return batchProt

