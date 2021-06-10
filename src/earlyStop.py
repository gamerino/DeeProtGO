# ==============================================================================
# IBB-CONICET-UNER
# sinc(i)-CONICET-UNL. http://sinc.unl.edu.ar/
# G. Merino, et al.
# gmerino@ingenieria.uner.edu.ar
# gmerino@sinc.unl.edu.ar
# ==========================================================================

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience = 10, verbose = False, delta = 0, res_dir = './', modelName = 'model' ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            res_dir (str): Path to the directory where trained model will be saved.
                            Default: './'
            modelName (str): File name where the model will be saved.
                            Default: 'model'                       
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.resDir = res_dir
        self.modelName = modelName
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: %d out of %.6f'%(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased (%.6f --> %.6f).  Saving model ...'%(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.resDir + self.modelName + 'checkpoint.pt')
        self.val_loss_min = val_loss
