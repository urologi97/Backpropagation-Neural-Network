import numpy as np
from OOP.Aktifasi import *
class Backprop:
    def __init__(self,alpha):
        self.alpha=alpha

    def DistribusiErrorOutput(self,dataOutput, kelas):
        sigmoid_turunan = Aktifasi()
        turunan = sigmoid_turunan.TurunanAktivasi(dataOutput)
        temp = kelas - dataOutput
        return temp * turunan

    def DistribusiError(self,faktorerror, dataOutput): #Jika hidden neuron lebih dari 1
        sigmoid_turunan = Aktifasi()
        turunan = sigmoid_turunan.TurunanAktivasi(dataOutput)
        return faktorerror * turunan

    def DeltaBobot(self,error, inputNeuron):
        bobot = []
        for x in range(len(inputNeuron)):
            temp = []
            for y in range(len(error[0])):
                BobotNeuron=(np.dot(error[x][y],inputNeuron[x]))*self.alpha
                temp.append(BobotNeuron)
            bobot.append(temp)
        return bobot

    def DeltaBias(self,error):
        return error * self.alpha

    def FaktorError(self,BobotSekarang, error):
        # return np.dot(np.transpose(BobotSekarang), np.transpose(error))
        return np.dot(BobotSekarang,np.transpose(error))