# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:11:34 2018

@author: reza
"""

import keras
from keras.layers import *
from keras.models import *
import keras.backend as K
import timeit
import numpy as np
#################### LONG SHORT TERM MEMORY (LSTM) DECODER ##########################

class Full_Model_LSTMDecoder(object):

    """
    Class for Full model LSTM decoder


    """

    def __init__(self,verbose=0,path='PositionEstimatorFull.h5'):
    
         self.verbose=verbose
         self.path=path

    def penalized_loss(self,Pen):
        # Define cost function for bouth positions
        Pen1=(K.abs(Pen-0.5)+.1)*2
        def loss(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true)*Pen1 , axis=-1)
        return loss

    def create_model(self,X_train,y_train):
        

        """
        Create LSTM Decoder
       
        """
        
        # Create position estimator model
        In=Input(shape=(X_train.shape[1:]),name='Input')
        Maze=LSTM(4,return_sequences=True,name='MazeSeq')(In)
        Ve=LSTM(2,return_sequences=True,name='VelocitySeq')(In)
        V=TimeDistributed(Dense(2,activation='sigmoid'),name='Velocity')(Ve)
        M=concatenate([Maze,In,Ve],name='ConcatFeature')
        Pe=LSTM(2,return_sequences=True,name='PositionSeq')(M)
        Maze=TimeDistributed(Dense(4,activation='softmax'),name='Maze')(Maze)
        P=TimeDistributed(Dense(2,activation='sigmoid'),name='Position')(Pe)
        Model2=Model(inputs=In,outputs=[P,V,Maze])


        
        Model2.compile(loss={'Velocity':'mse','Position':self.penalized_loss(V),'Maze':'binary_crossentropy'},
            optimizer='rmsprop',metrics={'Maze':'accuracy'})
        Model2.summary()
        self.model=Model2
        
    
    
    def fit(self,X_train,y_train,Arm_train,save=False,use_pretrained=False):
        ## Fit the model    
        if use_pretrained == True:
            self.model.load_weights(self.path)
            
        Hist=self.model.fit(X_train,[y_train[:,:,:2],y_train[:,:,2:],Arm_train],shuffle=False,epochs=400,verbose=self.verbose,batch_size=1)
        
        if save == True:
            self.model.save_weights(self.path)
        return Hist
    
    def save_or_load(self,save=False,use_pretrained=False):
        if use_pretrained == True:
            self.model.load_weights(self.path)
        if save == True:
            self.model.save_weights(self.path)
            
    def predict(self,X_test):
        # Create position estimator model
        In=Input(shape=(X_test.shape[1:]),name='Input')
        Maze=LSTM(4,return_sequences=True,name='MazeSeq')(In)
        Ve=LSTM(2,return_sequences=True,name='VelocitySeq')(In)
        V=TimeDistributed(Dense(2,activation='sigmoid'),name='Velocity')(Ve)
        M=concatenate([Maze,In,Ve],name='ConcatFeature')
        Pe=LSTM(2,return_sequences=True,name='PositionSeq')(M)
        Maze=TimeDistributed(Dense(4,activation='softmax'),name='Maze')(Maze)
        P=TimeDistributed(Dense(2,activation='sigmoid'),name='Position')(Pe)
        Model2=Model(inputs=In,outputs=[P,V,Maze])



        Model2.compile(loss={'Velocity':'mse','Position':'mse','Maze':'binary_crossentropy'},
                   optimizer='rmsprop',metrics={'Maze':'accuracy'})


    # Create summary of model
        Model2.summary()
        Model2.load_weights(self.path)
        y_predict=np.zeros((X_test.shape[1],4))
        start = timeit.default_timer()

        [y_valid_predicted_lstm,Vs,Maze]=Model2.predict(X_test)

        y_predict[:,0]=y_valid_predicted_lstm[0,:,0]
        y_predict[:,1]=y_valid_predicted_lstm[0,:,1]
        y_predict[:,2]=Vs[0,:,0]
        y_predict[:,3]=Vs[0,:,1]

        stop = timeit.default_timer()
        print('test time=%f'% (stop - start) )
        return [y_predict,Maze]


