'''
Created on 04/10/2017

@author: robotica
'''

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input, Concatenate, Softmax 
from keras.optimizers import Adam, SGD
from keras import regularizers

from keras.utils import plot_model


from model.DendralNeuron import DendralNeuron

from plot.laplotter import LossAccPlotter

import numpy as np

def build_multi_flow( input_shapes, num_classes, LR ):
    
    # input_shapes[0] // images
    # input_shapes[1] // features
    
    #Convolutional
    input_conv = Input( shape = input_shapes[0])
    input_features = Input (shape = input_shapes[1])
    
    model_conv = Conv2D(6, kernel_size=(3, 3), activation='relu')(input_conv)
    
    model_conv = MaxPooling2D(pool_size=(2, 2))(model_conv)   
    
    model_conv = Conv2D(16, (3, 3), activation='relu')(model_conv) 
    
    model_conv = MaxPooling2D(pool_size=(2, 2))(model_conv)

    model_conv = Dropout(0.50)(model_conv)     

    model_conv = Flatten()(model_conv)

    merged = Concatenate()([model_conv,input_features])
    
    model = Dense(100, activation='relu')(merged)
    
    model = Dense(num_classes, activation='relu')(model)
    
    model = Softmax()(model)
  
    model_keras = Model(inputs=[input_conv,input_features], outputs= model)  
    
    #print(model_keras.summary())
    
    model_keras.compile(loss = "categorical_crossentropy",
                  optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    
    #plot_model(model_keras, to_file='/home/gerardo/Documents/workspace/Hybrid_MNN/Figsmodel.png', show_shapes=True)
    
    return model_keras



def build_custom_vgg_16( num_classes, LR, plot  ):
    
    from keras.applications import VGG16
    #import pydot
    #pydot.find_graphviz = lambda: True
    #from keras.utils import plot_model
    
    
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))    
    
    model =  Sequential()
   
    num_layers =  len(vgg16_model.layers)
    
    for layer in vgg16_model.layers[ 0 : -4 ]:
        layer.trainable = False
        model.add(layer)
   
   
    #plot_model(vgg16_model, show_shapes=True, to_file='/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/VGG16_original.pdf')
    #plot_model(model      , show_shapes=True, to_file='/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/parsed_model.pdf')
    
    #model.add( vgg16_model )
    
    model.add (Flatten())
    model.add( Dense(100, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.001, l2=0.001 )))
    model.add( Dense(num_classes, activation='softmax'))
             
    model.compile(loss = "categorical_crossentropy",
                  optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    
    #############################
    
    #conv_base = VGG16(weights='imagenet', 
    #                  include_top=False,
    #                  input_shape=(150, 150, 3))
    
    #model = Sequential()
    #model.add( conv_base)
    #model.add( Flatten())
    #model.add( Dense(256, activation='relu'))
    #model.add( Dense(num_classes, activation='softmax'))
    
    return model


def train_VGG_16_Datagen (model, train_generator, validation_generator, batch_size, epochs,X_train, y_train, X_test, y_test, nb_verbose, steps_per_epoch):

    hist = model.fit_generator(train_generator.flow(X_train, y_train, batch_size=batch_size), 
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs, 
                        validation_data=validation_generator.flow(X_test, y_test, batch_size = batch_size ),
                        validation_steps=steps_per_epoch,
                        verbose =  nb_verbose, 
                        workers = 8)
    return hist


    
def build_vgg_16( num_classes, LR, plot  ):
    
    
    from keras.applications import VGG16
    
    vgg16_model = VGG16(weights='imagenet', 
                        include_top=False,
                        input_shape=(150, 150, 3))

    
    num_layers =  len(vgg16_model.layers)
    
    for layer in vgg16_model.layers[ 0 : num_layers ]:
        layer.trainable = False
        
    model =  Sequential()
    model.add( vgg16_model )
    
    model.add (Flatten())
    
    model.add( Dense(80, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.001, l2=0.001) ) ) 
    model.add( Dense(num_classes, activation='softmax'))
             
    model.compile(loss = "categorical_crossentropy",
                  optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    
    if( plot ):
        import pydot
        pydot.find_graphviz = lambda: True
        from keras.utils import plot_model
        plot_model(vgg16_model, show_shapes=True, to_file='/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/model_VGG16_bottom.pdf')
        plot_model(model, show_shapes=True, to_file='/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/model_VGG16_top.pdf')
    
    #############################
    
    #conv_base = VGG16(weights='imagenet', 
    #                  include_top=False,
    #                  input_shape=(150, 150, 3))
    
    #model = Sequential()
    #model.add( conv_base)
    #model.add( Flatten())
    #model.add( Dense(256, activation='relu'))
    #model.add( Dense(num_classes, activation='softmax'))
    
    return model
    
    
    
    
def classify_Iris():
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)

    return cnf_matrix, class_names 


def generate_hyperparam_grid_Dendral_1_layer( min_num_of_neuron_x_layer, max_num_of_neuron_x_layer, num_of_trials, min_LR, max_LR ):

    hnn_models = np.random.randint(low = min_num_of_neuron_x_layer, high= max_num_of_neuron_x_layer, size=num_of_trials)
    LR_arr = np.random.uniform(low = min_LR, high= max_LR, size=num_of_trials)
    #batch_size_arr = np.random.random_integers( low = min_batch_size, high= max_batch_size, size=num_of_trials)
    #nb_epoch_arr = np.random.random_integers (  low = min_epoch, high = max_epoch, size = num_of_trials)
    
    return hnn_models, LR_arr

def generate_hyperparam_grid( min_num_of_layers, max_num_of_layers, max_num_of_neuron_x_layer, num_of_trials, min_LR, max_LR ):

    num_of_layers = np.random.random_integers(low = min_num_of_layers, high = max_num_of_layers, size = num_of_trials )

    dnn_models = []
    neurons_per_layer = []

    for n_layer in num_of_layers :
        for layer in range( n_layer ):
            
                if layer== 0 :
                    neurons = np.random.random_integers( low = int(max_num_of_neuron_x_layer /2) , high = max_num_of_neuron_x_layer)
                    neurons_top = neurons
                else:
                    neurons = np.random.random_integers( low = int(neurons_top /2),  high = neurons_top )
                    if ( neurons == 0):
                        neurons =1
    
                    neurons_top = neurons
                     
                neurons_per_layer.append( neurons)
            
                
        dnn_models.append( neurons_per_layer )
        neurons_per_layer = []
    
    
    LR_arr = np.random.uniform(low= min_LR, high= max_LR, size=num_of_trials)
    #batch_size_arr = np.random.random_integers( low = min_batch_size, high= max_batch_size, size=num_of_trials)
    #nb_epoch_arr = np.random.random_integers (  low = min_epoch, high = max_epoch, size = num_of_trials)
    
    return dnn_models, LR_arr

def build_Cat_Dog_Conv_Net():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=( 224, 224, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
def build_train_CIFAR_10_ConvNet(input_shape, num_classes, LR, x_train, y_train, batch_size, epochs, x_test, y_test, plot_loss = True ):
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss = "categorical_crossentropy",
              optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = False,
          validation_data=(x_test, y_test))
    
    best_idx =  np.argmax( hist.history['val_acc'] ) 
    
    print("\t\t ---> Best : " + str(best_idx)+ "   Acc Train:  " + str( hist.history['acc'][best_idx]) + 
              " Val Acc: " + str(  hist.history['val_acc'][ best_idx ] )  + 
              " LR: " + str( LR ) + 
              " batch_size: " + str( batch_size ) + 
              " nb_epoch: " + str( epochs) + 
              " model_params: " + str( model.count_params()) )
    
    if ( plot_loss ):
        plotter = LossAccPlotter(  show_regressions=False)
            
        for i in range( len(hist.history['loss']) ):
            plotter.add_values( i ,
            loss_train= hist.history['loss'][i],
            loss_val  = hist.history['val_loss'][i],
            acc_train = hist.history['acc'][i],
            acc_val  =  hist.history['val_acc'][i],
            redraw=False, draw_regression = False)
             
    
        
        plotter.redraw()
        plotter.block()
        #plotter.save_plot("/home/robotica/workspace/MNN_SGD/Figs/B/single/no_dendral_batch/loss/it_"+ str(idx_model))
        plotter.close()
    
def build_train_CIFAR_10_Morph_ConvNet(input_shape, num_classes, LR, x_train, y_train, batch_size, epochs, x_test, y_test):
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    
    model.add(Dense(num_classes, activation='tanh'))
    model.add( DendralNeuron(num_classes, activation = 'tanh' ))      
    model.add( Activation('softmax') )
    
    #model.add(Dense(512))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    
    #model.add(Dense(num_classes))
    #model.add(Activation('softmax'))
    
    model.compile(loss = "categorical_crossentropy",
              optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = False,
          validation_data=(x_test, y_test))
    
    return hist, model
    #best_idx =  np.argmax( hist.history['val_acc'] ) 
    
    #print("\t\t ---> Best : " + str(best_idx)+ "   Acc Train:  " + str( hist.history['acc'][best_idx]) + 
    #          " Val Acc: " + str(  hist.history['val_acc'][ best_idx ] )  + 
    #          " LR: " + str( LR ) + 
    #          " batch_size: " + str( batch_size ) + 
    #          " nb_epoch: " + str( epochs) + 
    #          " model_params: " + str( model.count_params()) )
        
def build_train_MNIST_ConvNet( input_shape, num_classes, LR, x_train, y_train, batch_size, epochs, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape) )
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss = "categorical_crossentropy",
              optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = False,
          validation_data=(x_test, y_test))
    
    best_idx =  np.argmax( hist.history['val_acc'] ) 
    
    print("\t\t ---> Best : " + str(best_idx)+ "   Acc Train:  " + str( hist.history['acc'][best_idx]) + 
              " Val Acc: " + str(  hist.history['val_acc'][ best_idx ] )  + 
              " LR: " + str( LR ) + 
              " batch_size: " + str( batch_size ) + 
              " nb_epoch: " + str( epochs) + 
              " model_params: " + str( model.count_params()) )
    

def build_MLNN( morph_neurons, activation, input_shape, num_classes, LR):
    
    model = Sequential()

    model.add(DendralNeuron(morph_neurons, activation= activation, input_shape=input_shape))
    
    if ( num_classes > 2):
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(num_classes, activation='sigmoid'))
    
    model.compile(loss = "categorical_crossentropy",
                  optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
        
    return model

def train_MLNN(model,  x_train, y_train, batch_size, epochs, x_test, y_test, nb_verbose):
    
    hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = False,
          validation_data=(x_test, y_test))
    
    return hist
    
def build_LeNet (input_shape, num_classes, LR, regularizar ):
    
    
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(6, (5, 5), padding="same",    input_shape=input_shape ) )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(16, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(120, (5, 5), padding="same"))
    model.add(Activation("relu"))
    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(84, kernel_regularizer= regularizers.l1_l2(l1=0.001, l2=0.001 )) )
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
        

    model.compile(loss = "categorical_crossentropy",
                  optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    
    return model

def train_Lenet_Datagen (model, train_generator, validation_generator, batch_size, epochs,X_train, y_train, X_test, y_test, nb_verbose, steps_per_epoch):

    hist = model.fit_generator(train_generator.flow(X_train, y_train, batch_size=batch_size), 
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs, 
                        validation_data=validation_generator.flow(X_test, y_test, batch_size = batch_size ),
                        validation_steps=steps_per_epoch,
                        verbose =  nb_verbose, 
                        workers = 4)
    return hist

def train_LeNet (model,  x_train, y_train, batch_size, epochs, x_test, y_test, nb_verbose):
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose = nb_verbose,  validation_data=(x_test, y_test), shuffle = False)    
    return hist

def build_train_Morph_LeNet (input_shape, num_classes, LR, x_train, y_train, batch_size, epochs, x_test, y_test):
    
    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(32, kernel_size= (5, 5) , input_shape=input_shape, padding ="same" ))
    
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(64, kernel_size=(5, 5), padding ="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
   
    #model.add(Dense(500))
    model.add( DendralNeuron(num_classes, activation = 'tanh' )) 
    model.add(Activation("softmax"))

    # softmax classifier
    #model.add(Dense(num_classes))
    #model.add(Activation("softmax"))

    model.compile(loss = "categorical_crossentropy",
        optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = False,
          validation_data=(x_test, y_test))
    
    return hist, model
   
def build_train_Lineal_Morph_LeNet (input_shape, num_classes, LR, x_train, y_train, batch_size, epochs, x_test, y_test):
    
    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(32, kernel_size= (5, 5) , input_shape=input_shape, padding ="same" ))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(64, kernel_size=(5, 5), padding ="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
   
    model.add(Dense(num_classes, activation = 'tanh' ))
    model.add( DendralNeuron(num_classes, activation = 'tanh' )) 
    model.add(Activation("softmax"))

    model.compile(loss = "categorical_crossentropy",
        optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = False,
          validation_data=(x_test, y_test))
    
    return hist, model
   
def build_train_Morph_Lineal_Lenet(input_shape, num_classes, dendrites, LR, x_train, y_train, batch_size, epochs, x_test, y_test):
    # initialize the model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(32, kernel_size= (5, 5) , input_shape=input_shape, padding ="same" ))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(64, kernel_size=(5, 5), padding ="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())

    # OJO SE AÑADEN EL DOBLE DE NEURONAS DENDRALES  !!!!!!!!!!!!
    # OJO SE AÑADEN EL DOBLE DE NEURONAS DENDRALES  !!!!!!!!!!!!
       
    model.add( DendralNeuron(dendrites, activation = 'tanh' ))
    model.add( Dense(num_classes, activation = 'tanh' )) 
    model.add( Activation("softmax"))

    model.compile(loss = "categorical_crossentropy",
        optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = False,
          validation_data=(x_test, y_test))
    
    return hist, model
    
