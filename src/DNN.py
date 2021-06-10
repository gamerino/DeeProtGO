# ==============================================================================
# IBB-CONICET-UNER
# sinc(i)-CONICET-UNL. http://sinc.unl.edu.ar/
# G. Merino, et al.
# gmerino@ingenieria.uner.edu.ar
# gmerino@sinc.unl.edu.ar
# ==============================================================================

import numpy as np
import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class DNN(nn.Module):
    """DeeProtGO deep neural network."""
    def __init__(self, N1, N2, N3, N4, N_in_1, N_in_2 = 0, N_in_3 = 0, N_in_4 = 0,
                 N_in_5 = 0, pN1_1 = 0.7, pN1_2 = 0.35, pN2_1 = 0.7, pN2_2 = 0.35,
                 pN3_1 = 0.7, pN3_2 = 0.35, pN4_1 = 0.7, pN4_2 = 0.35, 
                 pN5_1 = 0.7, pN5_2 = 0.35, pNO_1 = 0.7, pNO_2 = 0.3, pDrop = 0.5,
                 activFunc = F.elu):
        """
        Args:
            N1 (int): Numer of neurons of the first output layer.
            N2 (int): Numer of neurons of the second output layer.
            N3 (int): Numer of neurons of the third output layer.
            N4 (int): Numer of neurons of the fourth output layer.
            N_in_1 (int): Input size of the first encoding sub-network.
            N_in_n (int): Input size of the n-th encoding sub-network, n = 2, 3, 4, 5.
                          Default = 0 (It means, only n-1 encoding sub-network)
            pNn_1 (float): Number between 0 and 1 indicating the proportion of the n-th encoding sub-network input size, 
                           used for defining the number of neurons in the first hidden layer of this network. It means, the
                           first hidden layer of the n-th encoding sub-network will have int(pNn_1*N_in_n) neurons = 1, 2,
                           3, 4, 5.
                          Default = 0.7    
            pNn_2 (float): Number between 0 and 1 indicating the proportion of the n-th encoding sub-network input size, 
                           used for defining the number of neurons in the second hidden layer of this network. It means, the
                           second hidden layer of the n-th encoding sub-network will have int(pNn_2*N_in_n) neurons = 1, 2,
                           3, 4, 5.
                          Default = 0.35  
            pNO_1 (float): Number between 0 and 1, used for defining the number of neurons in the first hidden layer of
                           the classification sub-network as a proportion of the length of the features vector, obtained 
                           by concatenating the output of encoding sub-networks.
                          Default = 0.7    
            pNO_2 (float): Number between 0 and 1, used for defining the number of neurons in the second hidden layer of
                           the classification sub-network as a proportion of the length of the features vector, obtained 
                           by concatenating the output of encoding sub-networks.
                          Default = 0.3
            pDrop (float): Number between 0 and 1 indicating the dropout probability.
                          Default = 0.5
            activFunc (torch function): PyTorch function of the selected activation function for hidden layers. 
                                        Default = F.elu            
        """
        super().__init__()
        inputList = np.array([N_in_1, N_in_2,N_in_3,N_in_4,N_in_5])
        prop = np.array([[pN1_1,pN1_2], [pN2_1,pN2_2], [pN3_1,pN3_2], [pN4_1,pN4_2], [pN5_1,pN5_2]])
        prop = prop[ inputList > 0, :]
        inputList = inputList[ inputList > 0 ]
        totalInputs = len(inputList) 
        # First encoding sub-network 
        self.hidden_one_1 = nn.Sequential(nn.Linear(inputList[0], int(prop[0, 0] * inputList[0])),
                                          nn.BatchNorm1d(int(prop[0, 0] * inputList[0])))
        self.hidden_two_1 = nn.Sequential(nn.Linear(int(prop[0, 0] * inputList[0]), int(prop[0, 1] * inputList[0])),
                                          nn.BatchNorm1d(int(prop[0, 1] * inputList[0])))
        if totalInputs > 1:
            # Second encoding sub-network 
            self.hidden_one_2 = nn.Sequential(nn.Linear(inputList[1], int(prop[1, 0] * inputList[1])),
                                              nn.BatchNorm1d(int(prop[1, 0] * inputList[1])))
            self.hidden_two_2 = nn.Sequential(nn.Linear(int(prop[1, 0] * inputList[1]), int(prop[1, 1] * inputList[1])),
                                              nn.BatchNorm1d(int(prop[1, 1] * inputList[1])))        
            if totalInputs > 2:
                # Third encoding sub-network 
                self.hidden_one_3 = nn.Sequential(nn.Linear(inputList[2], int(prop[2, 0] * inputList[2])),
                                                  nn.BatchNorm1d(int(prop[2, 0 ] * inputList[2])))
                self.hidden_two_3 = nn.Sequential(nn.Linear(int(prop[2, 0] * inputList[2]), 
                                                            int(prop[2, 1] * inputList[2])),
                                                  nn.BatchNorm1d(int(prop[2, 1] * inputList[2])))
                if totalInputs > 3:
                    # Fourth encoding sub-network 
                    self.hidden_one_4 = nn.Sequential(nn.Linear(inputList[3], int(prop[3, 0] * inputList[3]))
                                       , nn.BatchNorm1d(int(prop[3, 0] * inputList[3])))
                    self.hidden_two_4 = nn.Sequential(nn.Linear(int(prop[3, 0] * inputList[3]), 
                                                                int(prop[3, 1] * inputList[3])),
                                                      nn.BatchNorm1d(int(prop[3, 1] * inputList[3])))
                    if totalInputs > 4:
                        # Fifth encoding sub-network 
                        self.hidden_one_5 = nn.Sequential(nn.Linear(inputList[4], int(prop[4, 0] * inputList[4])),
                                                          nn.BatchNorm1d(int(prop[4, 0] * inputList[4])))
                        self.hidden_two_5 = nn.Sequential(nn.Linear(int(prop[4, 0] * inputList[4]),
                                                                    int(prop[4, 1] * inputList[4])),
                                                          nn.BatchNorm1d(int(prop[4,1]*inputList[4])))
        # Concatanting features from encoding sub-networks for being the input of classification sub-network
        sumInp = 0
        for i in range(totalInputs):
            sumInp += int(prop[i, 1] * inputList[i])
        self.hidden_three = nn.Sequential(nn.Linear(sumInp, int(pNO_1 * sumInp)), 
                                          nn.BatchNorm1d(int(pNO_1 * sumInp)))
        self.hidden_four = nn.Sequential(nn.Linear(int(pNO_1 * sumInp), int(pNO_2 * sumInp)),
                                         nn.BatchNorm1d(int(pNO_2 * sumInp)))
        if(N1 > 0):  
            self.hidden_N1 = nn.Sequential(nn.Linear(int(pNO_2 * sumInp), N1))
            self.N1Norm = nn.BatchNorm1d(N1)
            self.hidden_N2 = nn.Sequential(nn.Linear(N1, N2))
        else: # If the number of GO terms is too small, it could be enough with three output layers 
            self.hidden_N2 = nn.Sequential(nn.Linear(int(pNO_2 * sumInp), N2))
        self.N2Norm = nn.BatchNorm1d(N2)
        self.hidden_N3 = nn.Sequential(nn.Linear(N2, N3))
        self.N3Norm = nn.BatchNorm1d(N3)
        self.output = nn.Linear(N3, N4)
        self.dropout = nn.Dropout(pDrop)
        self.activFunc = activFunc
        self.inputList = inputList
        self.inputProp = prop
        self.N1 = N1

    def forward(self, inData1, inData2 = None, inData3 = None, inData4 = None, inData5 = None):        
        y1 = self.activFunc(self.hidden_one_1(inData1))
        y1 = self.dropout(y1)
        y1 = self.activFunc(self.hidden_two_1(y1))
        y = self.dropout(y1)
        if len(self.inputList) > 1:
            y2 = self.activFunc(self.hidden_one_2(inData2))
            y2 = self.dropout(y2)
            y2 = self.activFunc(self.hidden_two_2(y2))
            y2 = self.dropout(y2)
            y = torch.cat((y, y2), dim = 1)
            if len(self.inputList) > 2:
                y3 = self.activFunc(self.hidden_one_3(inData3))
                y3 = self.dropout(y3)
                y3 = self.activFunc(self.hidden_two_3(y3))
                y3 = self.dropout(y3)
                y = torch.cat((y,y3), dim = 1)
                if len(self.inputList) > 3:
                    y4 = self.activFunc(self.hidden_one_4(inData4))
                    y4 = self.dropout(y4)
                    y4 = self.activFunc(self.hidden_two_4(y4))
                    y4 = self.dropout(y4)
                    y = torch.cat((y, y4), dim = 1)
                    if len(self.inputList)>4:
                        y5 = self.activFunc(self.hidden_one_5(inData5))
                        y5 = self.dropout(y5)
                        y5 = self.activFunc(self.hidden_two_5(y5))
                        y5 = self.dropout(y5)
                        y = torch.cat((y,y5), dim = 1)
        y = self.activFunc(self.hidden_three(y))
        y = self.dropout(y)
        x = self.activFunc(self.hidden_four(y))
        x = self.dropout(x)
        if self.N1 > 0:
            x = self.hidden_N1(x)
            N1Out = torch.sigmoid(x)
            x = self.N1Norm(x)
            x = self.activFunc(x)
        x = self.hidden_N2(x)
        N2Out = torch.sigmoid(x)
        x = self.N2Norm(x)
        x = self.activFunc(x)
        x = self.hidden_N3(x)
        N3Out = torch.sigmoid(x)
        x = self.N3Norm(x)
        x = self.activFunc(x)
        N4Out = torch.sigmoid(self.output(x))
        if self.N1 > 0:
            return N1Out, N2Out, N3Out, N4Out
        else:
            return N2Out, N3Out, N4Out
 
