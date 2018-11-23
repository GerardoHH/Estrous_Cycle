'''
Created on 24/08/2017

@author: robotica
'''

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD

from keras import regularizers

from model.DendralNeuron import DendralNeuron

import time

    
def buildModel_DNN(nb_neurons, nb_classes, input_dim,  activation , binary_Classification = False,  print_Sumary = False ):

    model = Sequential()

    for idx in range( len( nb_neurons )):
        if ( idx ==  0):
            model.add( Dense( nb_neurons[idx] , input_dim = (input_dim), use_bias=True) )
            model.add( Activation( activation) )
        else :
            model.add( Dense( nb_neurons[idx] , use_bias = True ))
            model.add( Activation(activation) )

    if ( not binary_Classification  ):
        model.add(Dense( nb_classes , use_bias=True) )
        model.add(Activation('softmax') )
    else:
        model.add(Dense( 1, use_bias=True ) )
        model.add(Activation('sigmoid'))
        
    if ( print_Sumary ):
        model.summary()
    
    return model

def build_MLP_DN( nb_neurons, input_dim,  output_dim,  activation):

    model = Sequential()

    for idx in  range ( len(nb_neurons)):
        if ( idx ==  0):
            if (activation == None):
                model.add( Dense( nb_neurons[idx] , input_dim = input_dim, use_bias=True) )
                model.add( Activation( 'relu') )
            else:
                model.add( Dense( nb_neurons[idx] , input_dim = input_dim, use_bias=True ) )
                model.add( Activation( activation ) )
            
            model.add(Dropout(0.25))
        else:
            if (activation == None):
                model.add( Dense( nb_neurons[idx] , use_bias=True) )
                model.add( Activation( 'relu') )
            else:
                model.add( Dense( nb_neurons[idx] , use_bias=True) )
                model.add( Activation( activation ) )
            

    if ( output_dim == 1 ): # Solo añado una neurona dendral
            model.add( DendralNeuron(1, activation = activation ))
            
    else:   # Se añade una capa de dendral neurons  igual al numero de clases de salida
            model.add( DendralNeuron(output_dim, activation = activation ))        
            model.add( Activation('softmax') )
    
    return model
    

def build_HybridModel_DN_MLP( neurons, activation, input_shape, output_shape, path_save  ):
    
    model = Sequential()
    model.add(DendralNeuron(neurons, activation= activation, input_shape=input_shape))
    
    if ( output_shape > 2):
        model.add(Dense(output_shape, activation='softmax'))
    else:
        model.add(Dense(output_shape, activation='sigmoid'))

    #plot_model(model, to_file=path_save +"model.png")
    
    return model

def trainModel_DNN(model, P, T, Ptest, Ttest, LR, batch_size, nb_epoch,  binary_classification = False,  v_verbose = True ):
    
    #Ptest, Ttest, batch_size, nb_epoch, hist, train_time, LR, nb_neurons, loops 
    
    
    if ( not binary_classification ):
        model.compile(loss='categorical_crossentropy',
              optimizer=SGD( lr = LR ,  momentum= 0.9, nesterov=True ),     
              metrics=['accuracy'] )
    else:
        adam = Adam(lr= LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08 )
        model.compile(loss='binary_crossentropy',
              optimizer=adam,     
              metrics=['accuracy'] )
    
    init_time = time.time()
    
    hist = model.fit(P, T, validation_data=(Ptest,Ttest),
          batch_size=batch_size, epochs = nb_epoch, shuffle=True, verbose= v_verbose)

    end_time = time.time()
    
    train_time = end_time - init_time
    
    #model.save('./models/model_'+str(nb_classes) + '_class_'+str(loops)+'.h5', overwrite=True)

    return hist, train_time

def train_HybridModel_DN_MLP( model, LR, P, T, Ptest, Ttest, batch_size, nb_epoch, v_verbose, modelChkPnt = None ):
    
    if ( T.shape[1] > 2 ):
        model.compile(loss='categorical_crossentropy', 
                      optimizer =Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),   
                      metrics=['accuracy'] )
    else:
        model.compile(loss='binary_crossentropy', 
                      optimizer =Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),   
                      metrics=['accuracy'] )
    
    
    init_time = time.time()
    
    if ( modelChkPnt == None):
        hist = model.fit(P, T, validation_data=(Ptest,Ttest), batch_size=batch_size, epochs = nb_epoch,
                     shuffle=True, verbose= v_verbose)
    else:
        hist = model.fit(P, T, validation_data=(Ptest,Ttest), batch_size=batch_size, epochs = nb_epoch,
                     shuffle=True, verbose= v_verbose, callbacks=[modelChkPnt])
        
                     #shuffle=True, verbose= v_verbose, callbacks=[early_stopping_monitor])
    end_time = time.time()
    
    train_time = end_time - init_time
    
    #if ( save_model  ):
    #    model.save('./models/model_'+str(nb_classes) + '_class_'+str(loops)+'.h5', overwrite=True)

    return [hist, train_time]

def train_HybridModel( model, LR, P, T, Ptest, Ttest, batch_size, nb_epoch, v_verbose ):
    
    if ( T.shape[1] > 1 ):
        model.compile(loss='categorical_crossentropy', 
                      optimizer =Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),   
                      metrics=['accuracy'] )
    else:
        model.compile(loss='binary_crossentropy', 
                      optimizer =Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),   
                      metrics=['accuracy'] )
    
    
    init_time = time.time()
    
    hist = model.fit(P, T, validation_data=(Ptest,Ttest), batch_size=batch_size, epochs = nb_epoch,
                     shuffle=True, verbose= v_verbose)
                     #shuffle=True, verbose= v_verbose, callbacks=[early_stopping_monitor])
    end_time = time.time()
    
    train_time = end_time - init_time
    
    #if ( save_model  ):
    #    model.save('./models/model_'+str(nb_classes) + '_class_'+str(loops)+'.h5', overwrite=True)

    return [hist, train_time]
    
    
    
    