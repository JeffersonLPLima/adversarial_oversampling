import cv2
import numpy as np
import keras
from keras import backend as K
from keras.layers import Conv2D,Activation, BatchNormalization,AveragePooling2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import Callback,warnings, EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import gc

import glob
	

from collections import Counter



def one_hot_encoding(data):
	data_encoded = []
	#['N','L', 'R', 'A', 'V', 'pb', 'E']
	for i in range(len(data)):
		if(data[i]=='N'):
			data_encoded.append([1,0,0,0,0,0,0])
		if(data[i]=='L'):
			data_encoded.append([0,1,0,0,0,0,0])
		if(data[i]=='R'):
			data_encoded.append([0,0,1,0,0,0,0])
		if(data[i]=='A'):
			data_encoded.append([0,0,0,1,0,0,0])
		if(data[i]=='V'):
			data_encoded.append([0,0,0,0,1,0,0])
		if(data[i]=='pb'):
			data_encoded.append([0,0,0,0,0,1,0])
		if(data[i]=='E'):
			data_encoded.append([0,0,0,0,0,0,1])


	return np.array(data_encoded)

def one_hot_decoding(data):
	data_encoded = []
	#['N','L', 'R', 'A', 'V', 'pb', 'E']
	for i in range(len(data)):
		if(data[i][0]==1):
			data_encoded.append('N')
		if(data[i][1]==1):
			data_encoded.append('L')
		if(data[i][2]==1):
			data_encoded.append('R')
		if(data[i][3]==1):
			data_encoded.append('A')
		if(data[i][4]==1):
			data_encoded.append('V')
		if(data[i][5]==1):
			data_encoded.append('pb')
		if(data[i][6]==1):
			data_encoded.append('E')


	return np.array(data_encoded)



def model_cnn():

    model = Sequential()

    model.add(Conv2D(64, (3,3),strides = (1,1), input_shape = [112, 112, 1],kernel_initializer='he_uniform'))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='he_uniform'))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(AveragePooling2D(pool_size=(2, 2), strides= (2,2)))

    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model    


def eval_gan_samples():
    classes = ['N','L', 'R', 'A', 'V', 'pb', 'E']
    batch =36
    scores = []
    epochs =32
    Nclass = len(classes) 
    cvconfusion = np.zeros((Nclass,Nclass))
    cvscores = []
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    X_gan, y_gan = load_gan_samples(classe)
    print('data gan loaded from type ', classe)

    
    #use gans samples iteratively
    prop = [0.2, 0.4, 0.6, 0.8, 1]
    #prop = [1]
    for prop_i in prop: 
        n_gan_samples = int(X_gan.shape[0]*prop_i)
        X_train, y_train = np.concatenate((Xtrain, X_gan[:n_gan_samples]), axis=0), np.concatenate((ytrain, y_gan[:n_gan_samples]), axis=0)
        model = model_cnn()
        X_train, X_test = np.expand_dims(X_train,axis=3), np.expand_dims(Xtest,axis=3)
        y_train, y_test = y_train, ytest

        model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch)

        ypred = model.predict(X_test)
        ypred = np.argmax(ypred,axis=1)
        ytrue = np.argmax(y_test,axis=1)
        
        cvconfusion[:,:] = confusion_matrix(ytrue, ypred)
        
        F1 = np.zeros((Nclass,1))
        
        for i in range(Nclass):
            F1[i]=2*cvconfusion[i,i]/(np.sum(cvconfusion[i,:])+np.sum(cvconfusion[:,i]))
        scores.append(F1)
    
    return scores

def load_gan_samples(classe):

    dir_ = 'gen_samples_dir/'
    image_list = []
    types_ = classe
    X_train = []
    y_train = []
    img_rows = 112
    img_cols = 112
    for filename in glob.glob(dir_+str(types_)+'*.png'):
        X_train.append(cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (img_rows, img_cols)))
        if(types_ == 'E'):
            y_train.append([0,0,0,0,0,0,1])
        if(types_ == 'A'):
            y_train.append([0,0,0,1,0,0,0])
    
    return np.array(X_train), np.array(y_train)

def oversampling_eval():
    classes = ['N','L', 'R', 'A', 'V', 'pb', 'E']
    batch =12
    epochs = 20
    rep = 10
    Nclass = len(classes) 
    cvconfusion = np.zeros((Nclass,Nclass,1))
    cvscores = []       
    counter = 0
    from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
    
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    
    for r in range(rep):
        aa = 0
        model = model_cnn2()
        #ada = RandomOverSampler(random_state=42)
        #ada = ADASYN(random_state=42)
        ada = SMOTE(random_state=42)

        X_train, y_train = ada.fit_resample(X=X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2])), y=y_train)
        X_train = X_train.reshape((X_train.shape[0],112,112,1))

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose = 0)

        ypred = model.predict(X_test)
        ypred = np.argmax(ypred,axis=1)
        ytrue = np.argmax(y_test,axis=1)
 
        cvconfusion[:,:,counter] = confusion_matrix(ytrue, ypred)
        F1 = np.zeros((Nclass,1))
        for i in range(Nclass):
            F1[i]=2*cvconfusion[i,i,counter]/(np.sum(cvconfusion[i,:,counter])+np.sum(cvconfusion[:,i,counter]))
            print("F1 measure for {} rhythm: {:1.4f}".format(classes[i],F1[i,0]))            
            
            cvscores.append(np.mean(F1)* 100)
            print("Overall F1 measure: {:1.4f}".format(np.mean(F1)))  
            
    print("F1 report - Mean - std")
    print(np.mean(cvscores), np.std(cvscores))

    return model


def load_data():
	
	image_list = []
	types = ['N','L', 'R', 'A', 'V', 'pb', 'E']
	X_train = []
	y_train = []
	img_rows = 112
	img_cols = 112
	for type_ in types:
		for filename in glob.glob('dataset/'+str(type_)+'*.png'):
			X_train.append(cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (img_rows, img_cols)))
			y_train.append(type_)
	return np.array(X_train), np.array(y_train)


#print(Counter(load_data()[1]))
#input()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.60
sess = tf.Session(config=config)

#eval_gan_samples()
#oversampling_eval()
#
