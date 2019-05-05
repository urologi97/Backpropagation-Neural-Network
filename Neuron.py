import numpy as np

class Bobot_dan_bias:
    def __init__(self):
        pass

    def Generate_bobot(self,banyak_fitur,banyak_neuron):
        return np.random.uniform(low=-1, high=1, size=(banyak_fitur,banyak_neuron))

    def generate_bias(self,banyak_neuron):
        return np.random.uniform(low=0, high=1, size=(banyak_neuron))

    def generate_bobot_nguyen_widrow(self, input_neuron, hidden_neuron, output_neuron):
        bobot_lama = np.random.uniform(low=-0.5, high=0.5, size=(input_neuron, hidden_neuron))
        bobot_baru = np.random.uniform(low=-0.5, high=0.5, size=(input_neuron, hidden_neuron))
        faktor_skala = 0.7 * np.power(output_neuron, (1 / input_neuron))
        vj_lama = np.zeros(hidden_neuron)
        for i in range(hidden_neuron):
            for j in range(input_neuron):
                vj_lama[i] += np.power(bobot_lama[j][i], 2)
            vj_lama[i] = np.sqrt(vj_lama[i])

        for i in range(input_neuron):
            for j in range(hidden_neuron):
                bobot_baru[i][j] = (faktor_skala * bobot_lama[i][j]) / vj_lama[j]
        return bobot_baru

    def generate_bias_nguyen_widrow(self, input_neuron, hidden_neuron, output_neuron):
        faktor_skala = 0.7 * np.power(output_neuron, (1 / input_neuron))
        return np.random.uniform(low=-faktor_skala, high=faktor_skala, size=hidden_neuron)

    def UpdateBobot(self,bobotlama, deltabobot):
        print(type(deltabobot))
        deltabobot = np.asarray(deltabobot)
        return bobotlama+deltabobot

    def UpdateBias(self,biaslama, deltabias):
        return biaslama + deltabias

    def setBobot(self,bobotnow):
        self.bobot=bobotnow

    def getBobot(self):
        return self.bobot

    def setBias(self, biasnow):
        self.bias = biasnow

    def getBias(self):
        return self.bias

    def setOutput(self,outputnow):
        self.output=outputnow

    def getOutput(self):
        return self.output