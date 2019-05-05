from OOP.Aktifasi import *
import numpy as np
class FeedForward:
    def __init__(self):
        pass


    def FeedForward(self,train, bobot, bias):
        # L_temp = np.dot(train, np.transpose(bobot))
        L_temp=np.dot(train,bobot)
        temp = L_temp + bias
        sigmoid=Aktifasi()
        L_aktivasi = sigmoid.Aktivasi(temp)
        return L_aktivasi
