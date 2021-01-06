#!/usr/bin/env python
# coding: utf-8

# **NEU DATABASE:**
# 
# http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

# Descrição do problema: 
# 
# Esta rede neural convolucional foi criada para classificar 6 tipos de falhas superficiais em chapas de aço: *rolled-in scale* (RS), *patches* (Pa), *crazing* (Cr), *pitted surface* (PS), *inclusion* (In) e *scratches* (Sc). A base de dados consiste em 1800 imagens com 300 amostras de cada tipo mais comum de falhas superficiais com resolução de 200x200 pixels, criada e disponibilizada pela *Northeastern University*. 
# 
# A dificuldade de se classificar as imagens é devido à similaridades das falhas de uma mesma família de defeitos, como os tipo *rolled-in scale*, *crazing*, e *pitted surface*. Adicionalmente, a influencia da iluminação e mudanças no material, altera os valores das cores dos pixels.

# ![alt text](http://faculty.neu.edu.cn/yunhyan/Webpage%20for%20article/NEU%20surface%20defect%20database/Fig.1.jpg)

# # **INICIO DO ALGORITMO:**

# Intalação de Bibliotecas Para Google Coolab

# In[1]:


#!pip install tensorflow-gpu==2.1.0.alpha0 #only needed for google coolab notebook!
#!pip  install -q tf-nightly-2.0-preview
import zipfile
#from google.colab import drive #only needed for google coolab notebook!

import glob
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import misc
import cv2

import os.path as path
from scipy import misc
from PIL import Image
import random


# Bibliotecas Para Redes Neurais

# In[2]:



import datetime

#inicia tensorboard
get_ipython().run_line_magic('load_ext', 'tensorboard')

#remove todos logs de opreações anteriores
#!rm -rf ./logs/

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# **LIBERA ACESSO AO GOOGLE DRIVE**

# In[3]:


from google.colab import drive
drive.mount('/content/drive/', force_remount=True)


# Diretório de arquivos para treinamento:

# In[3]:


#Para Google Coolab:
#ZIP_PATH = '/content/drive/My Drive/Colab Notebooks/Data_Test/NEU - Steel Superficial Defects/NEU surface defect database.zip'
#IMAGE_PATH = '/content/drive/My Drive/Colab Notebooks/Data_Test/NEU - Steel Superficial Defects/TEMPORARIO/NEU surface defect database'

#Para windowns:
ZIP_PATH = 'C:/Users/JG/Desktop/RNA GUND/Codigo/'
IMAGE_PATH = 'C:/Users/GUND/Desktop/RNA GUND/Codigo/NEU surface defect database/'


# EXTRAI ARQUIVO COMPACTADO PARA PASTA DO GOOGLE DRIVE (É NECESSÁRIO EXTRAIR SOMENTE UMA VEZ!!)

# In[ ]:


get_ipython().system('mkdir IMAGE_PATH')

zip_ref = zipfile.ZipFile(ZIP_PATH, 'r')
zip_ref.extractall(IMAGE_PATH)
zip_ref.close()


# Abre as imagens descompactadas previamente:

# In[4]:


#Cria lista de arquivos cuja extensão seja "BMP"
file_paths = glob.glob(path.join(IMAGE_PATH, '*.bmp'))

num_imagens = len(file_paths)
num_imagens
#print(file_paths)


# IMPORTA AS IMAGENS CONTIDAS NA PASTA

# In[5]:


img_width  = 50
img_height = 50

images = [cv2.imread(path)   for path in file_paths]

imgs_resized = [cv2.resize(image, (img_width,img_height)) for image in images ]

images = np.asarray(imgs_resized)

image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])


# EXTRAI OS NOMES DAS IMAGENS PARA A CLASSIFICAÇÃO

# In[6]:


#Lê os nomes das figuras
n_images = images.shape[0]
y_classes =[]
y_img_names =[]
for i in range(n_images):
    filename = path.basename(file_paths[i])[0:-4]
    y_img_names.append(filename)
    filename = path.basename(file_paths[i])[0:2]
    y_classes.append(filename)

y_img_names[156]
y_classes[156]


# CALCULA O NÚMERO TOTAL DE CLASSES 

# In[7]:


# Scale
#X_data = images / 255

#num_train, height, width, depth = images.shape 
#num_test = X_test.shape[0] 
#Há 1800 imagens, com 6 classes de falhas, cada uma com 300 fotos
num_classes = np.unique(y_classes).shape[0] 

num_classes


# NORMALIZA TODOS OS VALORES DE CODIGOS DE CORES RGB PARA ESCALA DE 0..1

# In[8]:


X_data = images
X_data = X_data.astype('float32')
X_data = X_data / 255


X_data /= np.max(X_data)


# APLICA ENCODER ÀS CLASSES

# In[9]:


labelencoder = LabelEncoder()

y_classes = labelencoder.fit_transform(y_classes)

Y_classes_encoded = utils.to_categorical(y_classes, num_classes) # One-hot encode the labels


# In[ ]:


y_classes[1500:1510]


# In[ ]:


Y_classes_encoded[1500:1510]


# ### **Divide a base de dados entre teste e treinamento:**
# 
# 
# 
# 

# In[10]:


#test_size = percentual da base de dados destinado para testes
X_train, X_test, Y_train, Y_test = train_test_split(X_data , y_classes, test_size = 0.03, random_state = 0)
#X_train, X_test, Y_train, Y_test = train_test_split(X_data , y_classes, test_size = 0.1, random_state = 0)

Y_train_encoded = utils.to_categorical(Y_train, num_classes)
Y_test_encoded = utils.to_categorical(Y_test, num_classes)

#X_train, X_test, Y_train_encoded, Y_test_encoded = train_test_split(X_data , Y_classes_encoded, test_size = 0.03, random_state = 0)
#X_train, X_test, Y_train_encoded, Y_test_encoded = train_test_split(X_data , Y_classes_encoded, test_size = 0.03, random_state = 400)
print(Y_train[0:10])

n_training = len(X_train)
n_test = len(X_test)


# IMPRIME PARTE DA BASE DE DADOS A SER UTILIZADA PARA TREINAMENTO

# In[11]:


L_grid = 10
W_grid = 10

fig, axes = plt.subplots(L_grid, W_grid, figsize = (30,30))

axes = axes.ravel() #flatten the 28x28 image to 784 array

print("n_training = ", n_training)

index = []
#select random number from 0 to n_test

if n_training > W_grid * L_grid:
  index = random.sample(list(range(0, n_training)), (W_grid * L_grid))
  print(np.shape(index))
  print(index[0])
  for i in np.arange(0, W_grid * L_grid):
      axes[i].imshow(X_train[index[i]])
      #axes[i].set_title(y_img_names[index[i]], fontsize = 12)
      axes[i].set_title(Y_train_encoded[index[i]], fontsize = 12)
      axes[i].axis('off')
else:
  index = random.sample(list(range(0, n_training)), n_training)
  print(np.shape(index))
  print(index[0])
  for i in np.arange(0, n_training):
    axes[i].imshow(X_train[index[i]])
    #axes[i].set_title(y_img_names[index[i]], fontsize = 12)
    axes[i].set_title(Y_train_encoded[index[i]], fontsize = 12)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


# IMPRIME PARTE DA BASE DE DADOS A SER UTILIZADA PARA TESTES

# In[12]:


L_grid = 10
W_grid = 10

fig, axes = plt.subplots(L_grid, W_grid, figsize = (30,30))

axes = axes.ravel() #flatten the 28x28 image to 784 array

print("n_test = ", n_test)

index = []
#select random number from 0 to n_test



if n_test > W_grid * L_grid:

  index = random.sample(list(range(0, n_test)), (W_grid * L_grid))
  print(np.shape(index))
  print(index[0])

  for i in np.arange(0, W_grid * L_grid):
      axes[i].imshow(X_test[index[i]])
      #axes[i].set_title(y_img_names[index[i]], fontsize = 12)
      axes[i].set_title(Y_test_encoded[index[i]], fontsize = 12)
      axes[i].axis('off')
else:
  index = random.sample(list(range(0, n_test)), (n_test))
  print(np.shape(index))
  print(index[0])

  for i in np.arange(0, n_test):
    axes[i].imshow(X_test[index[i]])
    #axes[i].set_title(y_img_names[index[i]], fontsize = 12)
    axes[i].set_title(Y_test_encoded[index[i]], fontsize = 12)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


# INICIALIZAÇÃO DE VARIÁVEIS:
# 
# batch_size: determines the number of samples in each mini batch. Its maximum is the number of all samples, which makes gradient descent accurate, the loss will decrease towards the minimum if the learning rate is small enough, but iterations are slower. Its minimum is 1, resulting in stochastic gradient descent: Fast but the direction of the gradient step is based only on one example, the loss may jump around. batch_size allows to adjust between the two extremes: accurate gradient direction and fast iteration. Also, the maximum value for batch_size may be limited if your model + data set does not fit into the available (GPU) memory.
# 
# steps_per_epoch: the number of batch iterations before a training epoch is considered finished. If you have a training set of fixed size you can ignore it but it may be useful if you have a huge data set or if you are generating random data augmentations on the fly, i.e. if your training set has a (generated) infinite size. If you have the time to go through your whole training data set I recommend to skip this parameter.
# 
# 
# validation_steps: similar to steps_per_epoch but on the validation data set instead on the training data. If you have the time to go through your whole validation data set I recommend to skip this parameter.

# In[22]:



#numero de registros que irá calcular antes de atualizar os pesos (batch_size)
batch_size = 100
#num_epochs = 30
num_epochs = 40
kernel_size = (3,3)
pool_size = 2 

#conv_depth_1 e conv_depth_2 -> numero de detectores (mapas) de características (kernels)
conv_depth_1 = 32
conv_depth_2 = 64

#https://timodenk.com/blog/tensorflow-batch-normalization/
drop_prob_1 = 0.5
drop_prob_2 = 0.75
drop_prob_3 = 0.7


Pooling_size_1 = (2,2)
Pooling_size_2 = (2,2)


#Chute inicial para a quantidade de neurônios: 
# ((img_width  - kernel_size + 1) / Pooling_size) ^ 2 -> ((50 - 3 + 1) / 2) ^ 2 = 576

#hidden_neurons_1 = 260
#hidden_neurons_2 = 260

hidden_neurons_1 = 128
hidden_neurons_2 = 128


# ** AVALIAÇÃO DE MELHORES PARÂMETROS PARA A ARQUITETURA DA REDE NEURAL**

# In[12]:


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# Cria Nova Rede Com Parâmetros Para Fazer Validação Cruzada:

# In[74]:



# Define the Keras TensorBoard callback.
#logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir="logs/fit/" + "CrossValidation"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[13]:


def neural_net_steel_defects_tunning(n_dense_layers, neurons):
  
    classificador = Sequential()
 
    #1 camada de convolucao
    classificador.add(Conv2D(conv_depth_1, kernel_size, input_shape=(img_width,img_height,3),  padding='same', activation='relu'))
    #classificador.add(BatchNormalization())
    classificador.add(Conv2D(conv_depth_1, kernel_size, input_shape=(img_width,img_height,3),  padding='same', activation='relu'))
    #classificador.add(BatchNormalization())
    #Adiciona normalização à camada para aumentar a eficiencia e velocidade de processamento
    #classificador.add(BatchNormalization())
    #Aplica janela de pooling dos pixels, de tamanho 2x2
    classificador.add(MaxPooling2D(pool_size = Pooling_size_1 ))
    #dropout de 20% para ignorar pixels aleatorios da imagem visando reduzir a contribuição de pixels 
    #que não contribuem de fato com características das falhas a serem processadas
    classificador.add(Dropout(drop_prob_1))

    #2 camada de convolucao
    classificador.add(Conv2D(conv_depth_2, kernel_size, input_shape=(img_width,img_height,3),  padding='same', activation='relu'))
    #classificador.add(BatchNormalization())
    classificador.add(Conv2D(conv_depth_2, kernel_size, input_shape=(img_width,img_height,3),  padding='same', activation='relu'))
    #classificador.add(BatchNormalization())
    #Adiciona normalização à camada para aumentar a eficiencia e velocidade de processamento
    #classificador.add(BatchNormalization())
    #Aplica janela de pooling dos pixels, de tamanho 2x2
    classificador.add(MaxPooling2D(pool_size = Pooling_size_1 ))
    #dropout de 20% para ignorar pixels aleatorios da imagem visando reduzir a contribuição de pixels 
    #que não contribuem de fato com características das falhas a serem processadas
    classificador.add(Dropout(drop_prob_2))
    
    #classificador.add(BatchNormalization())
    
    classificador.add(Flatten()) 
    

    #Adiciona N camadas, para testar qual é a quantidade que causa menor erro 
    for i in range(1, n_dense_layers):
      classificador.add(Dense(units = neurons, activation='relu'))
      classificador.add(Dropout(drop_prob_3))


    classificador.add(Dense(units = num_classes, activation='softmax'))
    

    classificador.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return classificador


# In[14]:


classificador_grid = KerasClassifier(build_fn = neural_net_steel_defects_tunning)


# In[15]:


#Cria um "dicionario" de parametros a serem testados
#parametros_grid = { 'batch_size': [32, 100],
#                   'epochs': [30, 50, 100],
#                    'neurons': [128, 256, 576],
#                    'n_dense_layers': [1, 2 ,3]}

parametros_grid = { 'batch_size': [100],
                    'epochs': [40],
                    'neurons': [128, 260, 576],
                    'n_dense_layers': [1, 2, 3]}
                    


# In[16]:


grid_search = GridSearchCV(estimator = classificador_grid, param_grid = parametros_grid, scoring = 'accuracy', cv = 5 )
#grid_search = GridSearchCV(estimator = classificador_grid, param_grid = parametros_grid, scoring = 'accuracy' )


# In[17]:


y_classes
n_y = list(y_classes).count(0) #contagem de quantas vezes há a ocorrencia da classe 3...1..5..
n_y


# In[18]:


grid_search = grid_search.fit(X_data, y_classes)
#grid_search = grid_search.fit(X_data, y_classes, validation_split=0.1)


# In[19]:


melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_


# In[20]:


melhores_parametros 


# In[21]:


melhor_precisao


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard.notebook')
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# ### **Cria a Estrutura da Rede Neural Baseado Nos Parâmetros "Ótimos":**
# 
# 
# 

# In[23]:


def criar_rede_steel_defects():
    classificador = Sequential()

    classificador = Sequential()

    #1 camada de convolucao
    classificador.add(Conv2D(conv_depth_1, kernel_size, input_shape=(img_width,img_height,3),  padding='same', activation='relu'))
    #classificador.add(BatchNormalization())
    classificador.add(Conv2D(conv_depth_1, kernel_size, input_shape=(img_width,img_height,3),  padding='same', activation='relu'))
    #classificador.add(BatchNormalization())
    #Adiciona normalização à camada para aumentar a eficiencia e velocidade de processamento
    #classificador.add(BatchNormalization())
    #Aplica janela de pooling dos pixels, de tamanho 2x2
    classificador.add(MaxPooling2D(pool_size = Pooling_size_1 ))
    #dropout de 20% para ignorar pixels aleatorios da imagem visando reduzir a contribuição de pixels 
    #que não contribuem de fato com características das falhas a serem processadas
    classificador.add(Dropout(drop_prob_1))

    #2 camada de convolucao
    classificador.add(Conv2D(conv_depth_2, kernel_size, input_shape=(img_width,img_height,3),  padding='same', activation='relu'))
    #classificador.add(BatchNormalization())
    classificador.add(Conv2D(conv_depth_2, kernel_size, input_shape=(img_width,img_height,3),  padding='same', activation='relu'))
    #classificador.add(BatchNormalization())
    #Adiciona normalização à camada para aumentar a eficiencia e velocidade de processamento
    #classificador.add(BatchNormalization())
    #Aplica janela de pooling dos pixels, de tamanho 2x2
    classificador.add(MaxPooling2D(pool_size = Pooling_size_1 ))
    #dropout de 20% para ignorar pixels aleatorios da imagem visando reduzir a contribuição de pixels 
    #que não contribuem de fato com características das falhas a serem processadas
    classificador.add(Dropout(drop_prob_2))
    
    #classificador.add(BatchNormalization())
    
    classificador.add(Flatten()) 

    

    
    #1 Camada Oculta
    classificador.add(Dense(units = hidden_neurons_1, activation='relu'))
    #classificador.add(BatchNormalization())
    classificador.add(Dropout(drop_prob_2 ))
    
    

    '''
    #2 Camada Oculta
    classificador.add(Dense(units = hidden_neurons_2, activation='relu'))
    classificador.add(Dropout(drop_prob_2 ))
    
    #3 Camada Oculta
    classificador.add(Dense(units = hidden_neurons_2, activation='relu'))
    classificador.add(Dropout(drop_prob_2 ))
    '''

    classificador.add(Dense(units = num_classes, activation='softmax'))

    classificador.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return classificador

classificador_Steel_defects = criar_rede_steel_defects()


classificador_Steel_defects.summary()


# **AVALIAÇÂO DE PERFORMANCE DA REDE NEURAL COM A BASE DE DADOS**

# VALIDAÇÃO CRUZADA - KFOLD (DIVIDE A BASE DE DADOS EM PARTES ESPECIFICADAS E TREINA MULTIPLAS VEZES SEPARANDO UMA PARCELA PARA TREINO E OUTRA PARA TESTES, DE MODO A ENCONTRAR A COMBINAÇÃO DE PARTES QUE MINIMIZE O ERRO. ESTA PRATICA É AMPLAMENTE UTILIZADA NO MEIO CIENTÍFICO PARA ASSEGURAR QUE NENHUMA PARTE IMPORTANTE DA BASE DE DADOS SEJA IGNORADA E INFLUENCIE SIGNIFICATIVAMENTE NA VARIÂNCIA DO MODELO TREINADO)

# In[45]:


import tensorflow.keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[46]:


classificador_steel_def_KFOLD = KerasClassifier(build_fn=criar_rede_steel_defects, epochs = 40, batch_size = 100)


# In[47]:


#cv -> numero de vezes que executará o teste (implica também no numero de partes que será dividida a base de dados)
resultados = cross_val_score(estimator = classificador_steel_def_KFOLD, X = X_data, y = y_classes, cv = 10, scoring = 'accuracy')


# In[48]:


resultados


# In[49]:


resultados.mean()


# Quanto maior o valor do desvio padrão, mais overfitting há na rede neural

# In[50]:


resultados.std()


# In[ ]:


y_classes[0:10]


# # **SEM AUGUMENTATION:**

# Roda treinamento e em seguida aplica o teste, verificando o percentual de acerto (val_accuracy):
# 
# *quanto maior o "val_accuracy" e menor é o valor do "val_loss", obtidos a partir da base de dados de teste, melhor é a capacidade de generalização da rede

# In[24]:


epochs_hist = classificador_Steel_defects.fit(X_train, Y_train_encoded, steps_per_epoch = round(n_training / batch_size),
          batch_size=batch_size, epochs=num_epochs,
          validation_data=(X_test, Y_test_encoded))


# In[25]:


epochs_hist.history.keys()


# In[26]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Erro ao Longo do treinamento')
plt.xlabel('Época')
plt.ylabel('Erro')
plt.legend(['Erro durante Treino', 'Erro durante Testes'])


# In[27]:


plt.plot(epochs_hist.history['accuracy'])
plt.plot(epochs_hist.history['val_accuracy'])
plt.title('Acurácia ao Longo do treinamento')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(['Acurácia durante Treino', 'Acurácia durante Testes'])


# In[28]:


# get the predictions for the test data
predicted_classes = classificador_Steel_defects.predict_classes(X_test)


# In[29]:


L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # 

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i])
    axes[i].set_title("Predicao = {:0.1f}\n Verdadeiro = {}".format(predicted_classes[i], Y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 0.6)


# ### **COM "AUGUMENTATION": cria novas entradas de "imagens" a partir das existentes, rotacionando, "esticando", invertendo...**

# In[223]:


#For use in google coolab
# Define the Keras TensorBoard callback.
#logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#logdir="logs/fit/" + "Augumentation"



#For use in Windows?
'''
logs_base_dir = "./logs/fit/Augumentation"

os.makedirs(logs_base_dir, exist_ok=True)
os.join.path()

'''

logdir = './logs/fit/'


if not os.path.exists(logdir):
    os.mkdir(logdir)
dir_augumentation = os.path.join(logdir, "Augumentation")

#%tensorboard --logdir {logs_base_dir}
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=dir_augumentation, profile_batch = 100000000)


# In[32]:


gerador_treinamento = ImageDataGenerator(rotation_range= 7, horizontal_flip=True, shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)


# Augumentation com variação de luminosidade:

# In[33]:


gerador_treinamento = ImageDataGenerator(rotation_range= 7, horizontal_flip=True, shear_range=0.2, height_shift_range=0.07, zoom_range=0.2, brightness_range=[0.2,1.0])


# In[34]:


gerador_teste = ImageDataGenerator()


# In[35]:


#base_treinamento = gerador_treinamento.flow(X_train, Y_train_encoded, batch_size = batch_size )
gerador_treinamento.fit(X_train)
base_treinamento = gerador_treinamento.flow(X_train, Y_train_encoded, batch_size = batch_size )


# In[36]:


#base_teste = gerador_teste.flow(X_test, Y_test_encoded, batch_size = batch_size)
gerador_teste.fit(X_test)
base_teste = gerador_teste.flow(X_test, Y_test_encoded, batch_size = batch_size)


# In[38]:


#steps_per_epoch -> numero total de etapas/lotes de amostras a serem geradas pelo gerador antes de declarar uma época concluída. Coca-se a quantidade de imagens que temos, dividido pelo batch size
#classificador.fit_generator(base_treinamento, steps_per_epoch= 600000 / 128, epochs = 5, validation_data = base_teste, validation_steps= 10000 / 128)
#epochs_hist_augumentation = classificador_Steel_defects.fit_generator(base_treinamento, steps_per_epoch = num_imagens / batch_size, epochs = num_epochs, validation_data = base_teste, validation_steps= 10000 / batch_size)
#epochs_hist_augumentation = classificador_Steel_defects.fit_generator(base_treinamento, steps_per_epoch = num_imagens / 2, epochs = num_epochs, validation_data = base_teste, validation_steps= 1000 )
#epochs_hist_augumentation = classificador_Steel_defects.fit_generator(datagen.flow(X_train, Y_train_encoded, batch_size = batch_size), steps_per_epoch = n_training / batch_size, epochs = 2,  validation_data = base_teste)
epochs_hist_augumentation = classificador_Steel_defects.fit_generator(base_treinamento, steps_per_epoch = round(n_training / batch_size), epochs = 20,  validation_data = base_teste)


# com callback para tensorboard:

# In[ ]:


#steps_per_epoch -> numero total de etapas/lotes de amostras a serem geradas pelo gerador antes de declarar uma época concluída. Coca-se a quantidade de imagens que temos, dividido pelo batch size
epochs_hist_augumentation = classificador_Steel_defects.fit_generator(base_treinamento, steps_per_epoch = n_training / batch_size, epochs = 10,  validation_data = base_teste,  callbacks=[tensorboard_callback])


# In[ ]:


#Para google coolab:
#%load_ext tensorboard.notebook

#Para windows:
get_ipython().run_line_magic('load_ext', 'tensorboard')


get_ipython().run_line_magic('tensorboard', '--logdir dir_augumentation')



# In[39]:


epochs_hist_augumentation.history.keys()


# In[40]:


plt.plot(epochs_hist_augumentation.history['loss'])
plt.plot(epochs_hist_augumentation.history['val_loss'])
plt.title('Erro ao Longo do treinamento')
plt.xlabel('Época')
plt.ylabel('Erro de Treino e de Validação')
plt.legend(['Erro durante Treino', 'Erro durante Testes'])


# In[41]:


plt.plot(epochs_hist_augumentation.history['accuracy'])
plt.plot(epochs_hist_augumentation.history['val_accuracy'])
plt.title('Acurácia ao Longo do treinamento')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(['Acurácia durante Treino', 'Acurácia durante Testes'])


# In[80]:


# get the predictions for the test data
predicted_classes = classificador_Steel_defects.predict_classes(X_test)


# In[81]:


L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # 

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i])
    axes[i].set_title("Predicao = {:0.1f}\n Verdadeiro = {}".format(predicted_classes[i], Y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 0.6)


# In[ ]:




