from __future__ import print_function
from random import shuffle
import keras # Biblioteca para deep learning - Utilizando tensorflow como backend
import random
import glob
import os
import os.path
import sys
import numpy as np
# Concatenação na variavel de sistema pythonPath para suportar a biblioteca pandas
#sys.path.append("/usr/lib/python2.7/dist-packages")
#import pandas as pd
#
# Definição dos recursos utilizados pelo keras
from keras.preprocessing import sequence
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
# As proximas 2 classes (Tokenizer e pad_sequences) permitem que um corpo de textos seja vetorizado substituindo
# cada texto por uma sequencia de inteiros (representando o indice da palavra em um dicionario)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential,Model
from keras.layers import Input,Flatten,Dense,Dropout,GlobalAveragePooling2D,Conv2D,MaxPooling2D
from keras.utils.vis_utils import plot_model


from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Definição dos recursos utilizados pelo sklearn
from sklearn.datasets import load_files
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
#from sklearn.cross_validation import KFold

# Definiçao dos recursos do matplotlib - Utilizado para gerar gráficos
#import matplotlib.pyplot as plt
#import matplotlib.cm as colormap

import timeit

class RNC_vin:
    def __init__(self, max_features=100, embedding_dims=50, maxlen=50, filters=100, kernel_size=3, hidden_dims=250, output_dims=10, compile_loss='categorical_crossentropy'):
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.maxlen = maxlen
        self.filters = filters
        self.kernel_size = kernel_size
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        # model armazena o objeto referente a chamada Sequential() onde os parâmetros serão incluidos abaixo
        self.model = Sequential()
        
        # Adiciona uma camada de Embedding ao modelo de rede
        # max_features (input_dim) indica o tamanho do vocábulario
        # embedding_dims (output_dim) indica o tamanho do vetor espacial onde as palavras (numeros) serão armazenadas no embedding
        # input_length indica o tamanho maximo dos vetores que representam as frases (já tokenizadas)
        # Mais sobre Word Embedding: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
        self.model.add(Embedding(max_features,embedding_dims,input_length=maxlen))

        # Dropout é uma maneira de diminuir os erros durante a classificação
        # Basicamente, durante o treinamento, uma porcentagem dos neuronios são desligados aumentando a generalização da rede
        # forçando as camadas a aprender com diferentes neurônios
        # O parâmetro 0.2 é a fração que corresponde a quantidade de neurônios à desligar
        self.model.add(Dropout(0.2))

        # Adiciona uma camada de convolução na rede neural
        # filters representa a quantidade de convuluções, ou seja, a quantidade de saidas da camada Conv1D
        # kernel_size indica o tamanho de uma convolução (no caso de 3 em 3 - janelinha de 3x3)
        # padding=valid > diz para não realizar padding
        # activation possui a intenção de não linearizar a rede. reLU = Rectified Linear Units
        # strides controla como o filtro realiza uma convolução, 1 indica que o filtro sempre anda de uma em uma posição
        # mais sobre: https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
        self.model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))

        # Serve para reduzir o tamanho da representação da saída da camada de convolução acima, diminuindo o numero de parâmetros
        # e poder computacional exigido, consequentemente controlando o overfitting não perdendo as caracteristicas adquiridas nas camadas passadas
        # Nesse caso, ela usa a omeração de MAX() mara reduzir o tamanho dos dados
        self.model.add(GlobalMaxPooling1D())


        # Adiciona uma vanilla hidden layer - Vanilla indica que a camada é da forma mais simples, um padrão:
        # Dense aplica uma operação linear no vetor de entrada da camada
        # hidden_dims representa a dimensionalidade do array de saida
        self.model.add(Dense(hidden_dims))
        #dropou igual ao explicado acima
        self.model.add(Dropout(0.2))
        # Aplica uma função de ativação (no caso reLU) a uma saída
        self.model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid: - Modificacao
        # Transforma todos os neuronios em apenas 1 - MOD Classificacoes nao binarias
        self.model.add(Dense(output_dims)) #era 10!!!!

        # A partir do neuronio anterior, a funcao sigmoid gera valores entre 0 e 1
        # No caso, 0 representa 100% de certeza de ser negativo e 1 100% de certeza de ser Positivo
        self.model.add(Activation('sigmoid'))

        # Antes de treinar a rede, ela precisa ser compilada configurando o processo de aprendizado
        # STANCE - loss: Representa o objetivo no qual o modelo vai tentar minimizar binary_crossentropy é indicado para a classificação de 2 classes (positivo e negativo)
        # optimizer=adam é um algoritmo para atualizar os pesos da rede iterativamente baseado nos dados do treinamento
        # metrics para qualquer problema de classificação a métrica deve ser accuracy
        #self.model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics = ['accuracy'])
        self.model.compile(loss=compile_loss,optimizer='rmsprop',metrics = ['accuracy'])
        
        
    def fit(self, X_data, Y_data, batch_size=32, epochs=10, verbose=0, validation_data=None):
        if validation_data is not None:
            return self.model.fit(np.array(X_data), np.array(Y_data), batch_size=batch_size, epochs=epochs, validation_data=validation_data, verbose=verbose)
        else:
            return self.model.fit(np.array(X_data), np.array(Y_data), batch_size=batch_size, epochs=epochs, verbose=verbose)
        
    def predict(self, X_data, batch_size=32, verbose=0):
        return self.model.predict(X_data, batch_size=batch_size, verbose=verbose)
np.random.seed(1)
