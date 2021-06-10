# ==============================================================================
# IBB-CONICET-UNER
# sinc(i)-CONICET-UNL. http://sinc.unl.edu.ar/
# G. Merino, et al.
# gmerino@ingenieria.uner.edu.ar
# gmerino@sinc.unl.edu.ar
# ==========================================================================

import torch.nn as nn
import torch
import os, time, torch, pickle
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from src.DNN import DNN
import torch.nn.functional as F

class DNNModel:
    """DeeProtGO model. Initialization, training and testing functions."""
    def __init__(self, out_dir, nbatch, propOutFile, N_in_1, N_in_2 = 0, N_in_3 = 0, N_in_4 = 0, N_in_5 = 0,
                 pN1_1 = 0.7, pN1_2 = 0.35, pN2_1 = 0.7, pN2_2 = 0.35, pN3_1 = 0.7, pN3_2 = 0.35,
                 pN4_1 = 0.7, pN4_2 = 0.35,pN5_1 = 0.7, pN5_2 = 0.35, pNO_1 = 0.7, pNO_2 = 0.3,
                 pDrop = 0.5, activFunc = F.elu, optimMethod = torch.optim.Adam, criterion = nn.BCELoss, 
                 learningRate = 0.005, thresh = 0.3, useGPU = True):
        """
        Args:
            out_dir (str): Path to the directory where training, validation and testing results will be saved.
            nbatch (int): Batch size.
            propOutFile (str): Full path and name of the file were GO terms child/parents releationships are stored.
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
                           the classi
       	fication sub-network as a proportion of the length of the features vector, obtained 
                           by concatenating the output of encoding sub-networks.
                          Default = 0.3
            pDrop (float): Number between 0 and 1 indicating the dropout probability.
                          Default = 0.5
            activFunc (torch function): PyTorch function of the selected activation function for hidden layers. 
                                        Default = F.elu            
            optimMethod (torch function): PyTorch function of the selected optimization method.
                                          Default = torch.optim.Adam
            criterion (torch function): PyTorch function of the selected criteria for loss measure.
                                        Default = nn.BCELoss
            learningRate (float): Floating number indicating the learning rate.
                                        Default = 0.005
            thresh (float): Number between 0 and 1 indicating the threshold used for computing precision and recall during
                            model validation. 
                            Default = 0.03
            useGPU (bool): Logical indicating if a GPU must be used for model training and evaluation.
                            Default = True                            
        """
        self.nbatch = nbatch
        self.propOut = np.loadtxt(propOutFile, delimiter = "\t", dtype = "int")
        propCum = np.sum(self.propOut, axis = 1)
        # propCum represents the amount of parents that each term has (1 means itself)
        q1, q2, q3, q4 = np.quantile(propCum, q = (0.25, 0.5, 0.75, 1))
        N4 = np.where(propCum > q3)[0].shape[0]
        N3 = np.where((propCum <= q3) & (propCum > q2))[0].shape[0]
        N2 = np.where((propCum <= q2) & (propCum > q1))[0].shape[0]
        N1 = np.where(propCum <= q1)[0].shape[0]
        if (q1 == q2) or (q2 == q3) or (q3 == q4):
            q1, q2, q3 = np.quantile(propCum, q = (0.33, 0.67, 1))
            N4 = np.where(propCum > q2)[0].shape[0]
            N3 = np.where((propCum <= q2) & (propCum > q1))[0].shape[0]
            N2 = np.where(propCum <= q1)[0].shape[0]                
            N1 = 0
        self.net = DNN(N1, N2, N3, N4, N_in_1 = N_in_1, N_in_2 = N_in_2, N_in_3 = N_in_3, N_in_4 = N_in_4, N_in_5 = N_in_5,
                 pN1_1 = pN1_1, pN1_2 = pN1_2, pN2_1 = pN2_1, pN2_2 = pN2_2, pN3_1 = pN3_1, pN3_2 = pN3_2,
                 pN4_1 = pN4_1, pN4_2 = pN4_2,pN5_1 = pN5_1, pN5_2 = pN5_2, pNO_1 = pNO_1, pNO_2 = pNO_2,
                 pDrop = pDrop, activFunc = activFunc)
        GPUavailab = useGPU
        if useGPU:
            GPUavailab = torch.cuda.is_available()
        self.GPU = GPUavailab
        if self.GPU:
            self.net.cuda()
        self.optimizer = optimMethod(self.net.parameters(), lr = learningRate)
        self.criterion = criterion()
        self.out_dir = out_dir
        self.threshold = thresh
        self.N_out = N1 + N2 + N3 + N4
        self.outIndex = dict()
        self.outIndex["N4"] = np.where(propCum > q3)[0]
       	self.outIndex["N3"] = np.where((propCum <= q3) & (propCum > q2))[0]
        self.outIndex["N2"] = np.where((propCum <= q2) & (propCum > q1))[0]
        self.outIndex["N1"] = np.where(propCum <= q1)[0]
        if N1 == 0:
            self.outIndex["N4"] = np.where(propCum > q2)[0]
            self.outIndex["N3"] = np.where((propCum <= q2) & (propCum > q1))[0]
            self.outIndex["N2"] = np.where(propCum <= q1)[0]
            self.outIndex["N1"] = None
        self.propOut = torch.from_numpy(self.propOut)
        if self.GPU:
            self.net.cuda()
    def train(self, geneTerms, inData1, inData2 = None, inData3 = None, inData4 = None, inData5 = None):
        """ Model training: It obtains network output for the input data and compares it with the desired output.
                            Retropropagation is carried out. A loss measure and an F-1 performance score are returned. 
        Args:
            geneTerms (torch): PyTorch tensor with desired network output. Proteins in rows, GO terms in columns. 
            inData1 (torch): PyTorch tensor with the first input data. Proteins in rows, features in columns.
            inDataN (torch): PyTorch tensor with the N-th input data. Proteins in rows, features in columns. N = 2, 
                              3, 4, 5.
                             Default = None 
        """
        self.net.train()
        self.optimizer.zero_grad()
        geneTerms = geneTerms.float()
        GOOut, loss =  self.predict(inData1, inData2, inData3, inData4, inData5, geneTerms,
                                    isTrain = True)
        cpu = not self.GPU 
        if self.GPU:
            geneTerms = geneTerms.cuda()
        f1 = self.get_error(ref = geneTerms.detach(), out = GOOut.detach(), cpu = cpu)[0]
        return loss.item(), f1
    def test(self, geneTerms, inData1, inData2 = None, inData3 = None, inData4 = None, inData5 = None,
            propPrediction = False, CAFAerror = False, maxMeas = "F-score"):       
        """ Model testing: It obtains network output for the input data and compares it with the desired output 
                            returning performance scores. 
        Args:
            geneTerms (torch): PyTorch tensor with desired network output. Proteins in rows, GO terms in columns. 
            inData1 (torch): PyTorch tensor with the first input data. Proteins in rows, features in columns.
            inDataN (torch): PyTorch tensor with the N-th input data. Proteins in rows, features in columns. N = 2, 
                              3, 4, 5.
                             Default = None 
            propPrediction (bool): Logical indicating if prediction scores should be propagated to ancestors. If it is
                                   True, ancestors will have the highest score among their childs. 
                                   Default = True
            CAFAerror (bool): Logical indicating if CAFA performance measures should be computed. Recomended for model
                              testing with benchmark data.
                              Default = False
            maxMeas (str): Character indicating which CAFA measure (F-score, Precision or Recall) should be analyzed for
                           defining the optimal threshold.
                           Default = "F-score"
        """
        self.net.eval()
        toReturn = []
        batchSize = geneTerms.shape[0]
        geneTerms = geneTerms.float()
        with torch.no_grad():
            GOOut, loss =  self.predict(inData1, inData2, inData3, inData4, inData5, geneTerms,
                                    isTrain = False)
            toReturn.append(loss.item())
            if self.GPU:
                geneTerms = geneTerms.cuda()
            if CAFAerror:
                f, p, r, t = self.get_CAFA_error(geneTerms.detach(), GOOut.detach(), 
                                                 propPrediction = propPrediction, maxMeas = maxMeas, 
                                                 cpu = True)
                toReturn.append(f)
                toReturn.append(p)
                toReturn.append(r)
                toReturn.append(t)
                self.threshold = t
            cpu = not self.GPU     
            f, p, r = self.get_error(ref = geneTerms.detach(), out = GOOut.detach(), propPrediction = propPrediction,
                                     cpu = cpu)
            toReturn.append(f) 
            toReturn.append(p)
            toReturn.append(r)
        return toReturn
    def get_error(self, ref, out, propPrediction = True, cpu = True):
        """ Assessment of model error. 
        Args:
            ref (torch): PyTorch tensor with desired network output. Proteins in rows, GO terms in columns. 
            out (torch): PyTorch tensor with the network output. Proteins in rows, features in columns. 
            propPrediction (bool): Logical indicating if prediction scores should be propagated to ancestors. If it is
                                   True, ancestors will have the highest score among their childs. 
                                   Default = True
            cpu (bool): Use or not CPU for computing performance measures.
                              Default = True
        """        
        refOut = ref.clone()
        refOut = refOut.int()
        if propPrediction:
            discOut = self.propAnnot(out, cpu)
        else:
            discOut = out.clone()
        if cpu:
            refOut = refOut.cpu()
        discOut = discOut >= self.threshold
        pre = 0
        rec = 0
        for i in range(refOut.shape[0]):
            batchSampleOut = discOut[i,]
            batchRef = refOut[i,] == 1
            tp = torch.sum(torch.logical_and(batchSampleOut, batchRef)).item() 
            tp_fp = torch.sum(batchSampleOut).item()
            tp_fn = torch.sum(batchRef).item()
            if tp_fn == 0: #no annotations
                rec_i = 1
            else:
                rec_i = tp/tp_fn
            if tp_fp == 0: # no predictions
                if tp_fn == 0: # no annotations
                    pre_i = 1
                else: 
                    pre_i = 0
            else:
                pre_i = tp/tp_fp
            pre += pre_i
            rec += rec_i
        rec = rec / ref.shape[0]
        pre = pre / ref.shape[0]
        Fscore = (2 * pre * rec) / (pre + rec)     
        return Fscore, pre, rec
    def get_CAFA_error(self, ref, out, propPrediction = True, cpu = True, maxMeas = "F-score"):
        """ Assessment of model error. 
        Args:
            ref (torch): PyTorch tensor with desired network output. Proteins in rows, GO terms in columns. 
            out (torch): PyTorch tensor with the network output. Proteins in rows, features in columns. 
            propPrediction (bool): Logical indicating if prediction scores should be propagated to ancestors. If it is
                                   True, ancestors will have the highest score among their childs. 
                                   Default = True
            cpu (bool): Use or not CPU for computing performance measures.
                              Default = True
            maxMeas (str): Character indicating which CAFA measure (F-score, Precision or Recall) should be analyzed for
                           defining the optimal threshold.
                           Default = "F-score"                              
        """ 
        if propPrediction:
            propOut = self.propAnnot(out, cpu)
        else:
            propOut = out.clone()
        refAnnot = ref.clone()
        if cpu:
            refAnnot = refAnnot.cpu()
            propOut = propOut.cpu()
        protWithAnnot = torch.sum(ref, dim = 1) > 0 # has annotations?
        if any(protWithAnnot):
            propOut = propOut[protWithAnnot]
            refAnnot = refAnnot[protWithAnnot]
            # CAFA measures
            avgPre = []
            avgRec = []
            avgFscore = []
            threshold = 1 - np.arange(0.0, 1.0, 0.01)
            for thresh in threshold:
                # analyze first those with predictions
                protWithPred = torch.sum(propOut >= thresh, dim = 1) > 0 # has prediction?
                if any(protWithPred):
                    refOut = refAnnot[protWithPred].clone().int()
                    posOut = propOut[protWithPred].clone()
                    discOut = posOut >= thresh
                    pre = 0
                    rec = 0
                    fsc = 0
                    for i in range(refOut.shape[0]): #by positive proteins
                        tp = torch.sum(torch.logical_and(refOut[i,], discOut[i,])).item()
                        tp_fp = torch.sum(discOut[i, ]).item()
                        tp_fn = torch.sum(refOut[i, ]).item()
                        pre += tp / tp_fp
                        rec += tp / tp_fn          
                    pre = pre/posOut.shape[0]
                    rec = rec / propOut.shape[0] 
                    if  rec > 0:
                        avgF = 2 * pre * rec / ( pre + rec )
                    else:
                        avgF = 0
                else: 
                    rec = 0
                    pre = 0
                    avgF = 0
                avgRec.append(rec)
                avgPre.append(pre)
                avgFscore.append(avgF)
            maxF = np.max(avgFscore)
            meanPre = avgPre[ np.argmax(avgFscore) ]
            meanRec = avgRec[ np.argmax(avgFscore) ]
            if maxMeas == "F-score":
                thresh = threshold[ np.argmax(avgFscore) ]
            else:
                if maxMeas == "Precision":
                    thresh = threshold[ np.argmax(avgPre) ]
                    maxF = avgFscore[ np.argmax(avgPre) ]
                    meanRec = avgRec[ np.argmax(avgPre) ]
                    meanPre = np.max( avgPre )
                else:
                    thresh = threshold[ np.argmax(avgRec) ]
                    maxF = avgFscore[ np.argmax(avgRec) ]
                    meanPre = avgPre[ np.argmax(avgRec) ]
                    meanRec = np.max( avgRec )
            return maxF, meanPre, meanRec, np.round(thresh, 2)    
    def predict(self, inData1, inData2 = None, inData3 = None, inData4 = None, inData5 = None, 
                geneTerms = None, isTrain = False):
        """ It obtains network predictions for the input data. If is required, it compares it with the desired output.
                            During training, retropropagation is carried out 
        Args:
            inData1 (torch): PyTorch tensor with the first input data. Proteins in rows, features in columns.
            inDataN (torch): PyTorch tensor with the N-th input data. Proteins in rows, features in columns. N = 2, 
                              3, 4, 5.
                             Default = None 
            geneTerms (torch): PyTorch tensor with desired network output. Proteins in rows, GO terms in columns. 
                               Default = None (It indicates only predictions will be obtained)
            isTrain (bool): Logical for indicating if predictions are part of the training process, if it is True,
                            retropropagation is carried out. 
        """
        if self.GPU:
            inData1 = inData1.cuda()
        if inData2 is not None:
            if self.GPU:
                inData2 = inData2.cuda()
            if inData3 is not None:
                if self.GPU:
                    inData3 = inData3.cuda()
                if inData4 is not None:
                    if self.GPU:
                        inData4 = inData4.cuda()
                    if inData5 is not None:
                        if self.GPU:
                            inData5 = inData5.cuda()
                        NetOut = self.net(inData1.float(), inData2.float(), inData3.float(),
                                          inData4.float(), inData5.float())
                    else:
                        NetOut = self.net(inData1.float(), inData2.float(), 
                                          inData3.float(), inData4.float())
                else: 
                    NetOut = self.net(inData1.float(), inData2.float(), inData3.float())
            else:
                NetOut = self.net(inData1.float(), inData2.float())
        else: 
            NetOut = self.net(inData1.float())
        if len(NetOut) == 4:
            N1Out, N2Out, N3Out, N4Out = NetOut
        else:
            N2Out, N3Out, N4Out = NetOut
        if geneTerms is not None:
            if self.GPU:
                geneTerms = geneTerms.cuda()
            lossN4 = self.criterion(N4Out, geneTerms[:, self.outIndex["N4"]])
            lossN3 = self.criterion(N3Out, geneTerms[:, self.outIndex["N3"]])
            lossN2 = self.criterion(N2Out, geneTerms[:, self.outIndex["N2"]])
            if np.all(self.outIndex["N1"] != None):
                lossN1 = self.criterion(N1Out, geneTerms[:, self.outIndex["N1"]])
                loss = lossN1 + lossN2 + lossN3 + lossN4
            else:
                    loss = lossN4 + lossN3 + lossN2
        else: 
            loss = None
        if isTrain:
                loss.backward()
                self.optimizer.step()
        GOOut = torch.zeros((inData1.shape[0], self.N_out)).float()
        if self.GPU:
            N4Out = N4Out.detach().cpu()
            N3Out = N3Out.detach().cpu()
            N2Out = N2Out.detach().cpu()
            if np.all(self.outIndex["N1"] != None):
                N1Out = N1Out.detach().cpu()
        GOOut[:, self.outIndex["N4"]] = N4Out
        GOOut[:, self.outIndex["N3"]] = N3Out
        GOOut[:, self.outIndex["N2"]] = N2Out
        if np.all(self.outIndex["N1"] != None):
             GOOut[:, self.outIndex["N1"]] = N1Out
        if self.GPU:
            GOOut = GOOut.cuda()
        return GOOut, loss
    def getPredictions(self, inData1, inData2 = None, inData3 = None, inData4 = None, inData5 = None):
        """ It obtains network predictions for the input data. 
        Args:
            inData1 (torch): PyTorch tensor with the first input data. Proteins in rows, features in columns.
            inDataN (torch): PyTorch tensor with the N-th input data. Proteins in rows, features in columns. N = 2, 
                              3, 4, 5.
                             Default = None t obtains network predictions for the input data. 
        """
        self.net.eval()
        with torch.no_grad():
            GOPred,_ = self.predict(inData1, inData2, inData3, inData4, inData5)
        return GOPred
    def propAnnot(self, outPred, cpu = True):
        """ Propagating network predictions thus ancestors will have the highest score among their childs. 
        Args:
            outPred (torch): PyTorch tensor with the networl predictions.
            cpu (bool): Use or not CPU for doing the propagation.
                              Default = True            
        """
        propOut = self.propOut
        if (not cpu) and self.GPU :
            propOut = propOut.cuda()
        else:
            outPred = outPred.cpu()
        summedPred = torch.sum(outPred, dim = 1) > 0
        if any(summedPred):
            posOut = outPred[summedPred].clone()
            for j in range(posOut.shape[1]): 
                vals, _ = torch.max(outPred * propOut[:,j], dim=1)
                posOut[:,j] = vals
        fullOut = outPred.clone()
        if any(summedPred):
            fullOut[summedPred] = posOut
        if cpu:
            fullOut = fullOut.cpu()
        return fullOut

