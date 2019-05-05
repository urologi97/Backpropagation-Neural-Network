from OOP.FeedForward import *
from OOP.Backprop import *
from OOP.Neuron import *
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def convert (x):
  if x == 1: return [1,0,0,0,0,0,0,0,0,0,0,0,0]
  elif x == 2: return [0,1,0,0,0,0,0,0,0,0,0,0,0]
  elif x == 3: return [0,0,1,0,0,0,0,0,0,0,0,0,0]
  elif x == 4: return [0,0,0,1,0,0,0,0,0,0,0,0,0]
  elif x == 5: return [0,0,0,0,1,0,0,0,0,0,0,0,0]
  elif x == 6: return [0,0,0,0,0,1,0,0,0,0,0,0,0]
  elif x == 7: return [0,0,0,0,0,0,1,0,0,0,0,0,0]
  elif x == 8: return [0,0,0,0,0,0,0,1,0,0,0,0,0]
  elif x == 9: return [0,0,0,0,0,0,0,0,1,0,0,0,0]
  elif x == 10: return [0,0,0,0,0,0,0,0,0,1,0,0,0]
  elif x == 11: return [0,0,0,0,0,0,0,0,0,0,1,0,0]
  elif x == 12: return [0,0,0,0,0,0,0,0,0,0,0,1,0]
  else: return [0,0,0,0,0,0,0,0,0,0,0,0,1]


def konvert_kelas(data):
    temp=masukan_data(data)
    konvert=[]
    for x in range(len(data)):
        konvert.append(convert (temp[x]))
    return konvert
def masukan_data(data):
    temp=[]
    for x in range(len(data)):
        temp.append(data.values[x])
    return temp

def Menghitung_Prediksi(Y_test):
    Prediksi=[]
    for x in range(len(Y_test)):
        temp=Y_test[x][0]
        counter=1
        for y in range(len(Y_test[0])):
            if(Y_test[x][y]>temp):
                temp=Y_test[x][y]
                counter=y+1
        Prediksi.append(counter)
    return Prediksi

X_train=pd.read_csv('Data Latih.csv')
X_test=pd.read_csv('Data Uji.csv')
X_train_kelas = X_train["class"]
X_test_kelas = X_test["class"]
X_train=X_train.drop(labels = ["class"],axis = 1)
X_test=X_test.drop(labels = ["class"],axis = 1)
kelas_training=np.asarray(konvert_kelas(X_train_kelas))
kelas_test=np.asarray(X_test_kelas)
train=np.asarray(masukan_data(X_train))
test=np.asarray(masukan_data(X_test))

alpha=0.001

#inisialisasi Bpbot dan bias awal
bobot1=Bobot_dan_bias()

bobot1.setBobot(bobot1.Generate_bobot(len(train[0]),35))#Banyak fitur x jumlah neuron
bobot1.setBias(bobot1.generate_bias(35)) #Generate Verse biasa
# bobot1.setBobot(bobot1.generate_bobot_nguyen_widrow(len(train[0]), 35, 13))
# bobot1.setBias(bobot1.generate_bias_nguyen_widrow(len(train[0]),35,13)) #_nguyen_widrow


bobot2=Bobot_dan_bias()
bobot2.setBobot(bobot2.Generate_bobot(35,13))#Banyak neuron sebelumnya x jumlah neuron
bobot2.setBias(bobot2.generate_bias(13))

# train
for x in range(5000):
    # FeedForward

    maju = FeedForward()
    bobot1.setOutput(maju.FeedForward(train, bobot1.getBobot(), bobot1.getBias()))

    bobot2.setOutput(maju.FeedForward(bobot1.getOutput(), bobot2.getBobot(), bobot2.getBias()))

    # Backprop
    # 1 Mundur ke Hidden layer
    mundur = Backprop(alpha)
    errorL2 = mundur.DistribusiErrorOutput(bobot2.getOutput(), kelas_training)

    delta_Bobot_l2 = mundur.DeltaBobot(errorL2, bobot1.getOutput())

    delta_bias_l2 = mundur.DeltaBias(errorL2)

    # 2 Mundur ke Input layer
    Faktor_error = mundur.FaktorError(bobot2.getBobot(), errorL2)
    Faktor_error = np.transpose(Faktor_error)
    errorL1 = mundur.DistribusiError(Faktor_error, bobot1.getOutput())

    delta_Bobot_l1 = mundur.DeltaBobot(errorL1, train)
    delta_bias_l1 = mundur.DeltaBias(errorL1)

    # Update Bobot
    for x in range(len(delta_Bobot_l1)):
        bobot1.setBobot(bobot1.UpdateBobot(bobot1.getBobot(), np.transpose(delta_Bobot_l1[x])))
        bobot1.setBias(bobot1.UpdateBias(bobot1.getBias(), delta_bias_l1[x]))

    for x in range(len(delta_Bobot_l2)):
        bobot2.setBobot(bobot2.UpdateBobot(bobot2.getBobot(), np.transpose(delta_Bobot_l2[x])))
        bobot2.setBias(bobot2.UpdateBias(bobot2.getBias(), delta_bias_l2[x]))


#Test
Test=FeedForward()
bobot1.setOutput(Test.FeedForward(test,bobot1.getBobot(),bobot1.getBias()))
bobot2.setOutput(Test.FeedForward(bobot1.getOutput(),bobot2.getBobot(),bobot2.getBias()))

# for x in range(len (bobot2.getOutput())):
#     print(bobot2.getOutput()[x])

prediksi=Menghitung_Prediksi(bobot2.getOutput())

akurasi = accuracy_score(kelas_test, prediksi)
print(akurasi*100,"%")