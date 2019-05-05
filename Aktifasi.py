import numpy as np
class Aktifasi:
    def __init__(self):
        pass

    def Aktivasi(self,H_init):
        # return (2 / (1 + np.exp(-H_init))) - 1  # bipolar
        return 1 / (1+np.exp(-H_init)) #sigmoid

    def TurunanAktivasi(self,dataOutput):
        return (dataOutput*(1-dataOutput)) #sigmoid
        # return ((1 + dataOutput) * (1 - dataOutput) / 2)  # bipolar