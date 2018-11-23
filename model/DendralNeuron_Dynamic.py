import numpy as np                  
import matplotlib.pyplot as plt
from keras.datasets import mnist    
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation  
from keras.utils import np_utils  

from keras import backend as K
from keras.engine.topology import Layer
from keras import optimizers
from keras import activations


class DendralNeuron_Dynamic(Layer):
    
    def __init__(self, units, dendrites, activation=None, **kwargs):
        self.Nd = units   #
        self.dendrites = dendrites
        self.activation = activations.get(activation)
        super(DendralNeuron_Dynamic, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wmin = self.add_weight(name='Wmin', 
                                      shape=(self.dendrites, input_shape[1]),
                                      #initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                                      initializer='uniform',
                                      trainable=True)
        
        self.Wmax = self.add_weight(name='Wmax', 
                                      shape=(self.dendrites, input_shape[1]),
                                      #initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                                      initializer='uniform',
                                      trainable=True)
        
        super(DendralNeuron_Dynamic, self).build(input_shape) 

    def call(self, x):
        Q = K.int_shape(x)[0]
        if Q is None: Q = 1
        X = K.repeat(x,self.dendrites)
        Wmin = K.permute_dimensions(K.repeat(self.Wmin, Q), (1,0,2))
        L1 = K.min(X - Wmin, axis=2)
        Wmax = K.permute_dimensions(K.repeat(self.Wmax, Q), (1,0,2))
        L2 = K.min(Wmax - X, axis=2)
        output = K.minimum(L1,L2)
        if self.activation is not None:
           output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.dendrites) 


def main():

    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("Dimnesiones de X_train =", X_train.shape)
    print("Dimensiones de y_train =", y_train.shape)
    plt.rcParams['figure.figsize'] = (7,7) # Hacer las figuras mas grandes
    
    # Pre-Procesamiento
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print("Dimensiones de X_train =", X_train.shape)
    print("Dimensiones de y_train =", X_test.shape)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    def DNN02():
       # 98.63%
       model = Sequential()
       model.add(Dense(512, activation='relu', input_shape=(784,)))
       model.add(Dropout(0.2))   
       model.add(Dense(512, activation='relu'))
       model.add(Dropout(0.2))
       model.add(Dense(10, activation='softmax'))
       return model
    
    def h00():
       # mas lento en tiempo de entrenamiento, mas lento de converger,  
       # epochs = 50, learning_rate = 0.01, learning_decay = 0.0, batch=128, 300unidades y tanh, -> 97.70%
       model = Sequential()
       model.add(DendralNeuron_Dynamic(300, activation='tanh', input_shape=(784,)))  # Si roto las cajas quizas pueda mejorar >97.7% Entrenar la rotacion al ultimo. 
       model.add(Dense(10, use_bias=True, activation='softmax'))
       return model
    
    def DNN00():  
       # epochs = 50, learning_rate = 0.001, learning_decay = 0.0, batch=128, tanh
       #    100unidades -> 97.76%
       #    200unidades -> 98.15%
       #    400unidades -> 98.41%
       #   1000unidades -> 98.50%
       model = Sequential()
       model.add(Dense(100, activation='tanh', input_shape=(784,)))
       model.add(Dense(10, use_bias=True, activation='softmax'))
       return model
    
    def MD00():
       # epochs = 50, learning_rate = 0.01, learning_decay = 0.0, batch=128, 300unidades y tanh, -> 83.28%
       model = Sequential()
       model.add(DendralNeuron(300, activation='tanh', input_shape=(784,)))
       model.add(DendralNeuron(10, activation='softmax'))
       return model
    
    #--------------------------------------------------------------------------------------------
    # habra otras arquitecturas hybridas que mejoren el desempeo > 99%???????????????????????????????????????????
    def MD01():
       # epochs = 50, learning_rate = 0.01, learning_decay = 0.0, batch=128, unidades y tanh, -> 73%
       model = Sequential()
       model.add(DendralNeuron(100, activation='tanh', input_shape=(784,)))
       model.add(DendralNeuron(50, activation='linear'))
       model.add(DendralNeuron(10, activation='softmax'))
       return model
    
    def h01():
       # epochs = 50, learning_rate = 0.01, learning_decay = 0.0, batch=128, unidades y tanh, -> %
       model = Sequential()
       model.add(Dense(200, activation='tanh', input_shape=(784,)))
       model.add(DendralNeuron(200, activation='linear'))
       model.add(Dense(10, activation='softmax'))
       return model
    
    epochs = 50
    learning_rate = 0.01
    learning_decay = 0.0
    batch=128
    
    # Modelo 
    model = h00()
    
    # Entrenamiento
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=learning_decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    h = model.fit(X_train, Y_train, batch_size=batch, epochs=epochs, verbose=True, validation_data=(X_test, Y_test))
    
    # Evaluar modelo
    epoch_max = np.argmax(h.history['val_acc'])
    plt.plot(h.history['val_acc'],label='val_acc')
    plt.plot(h.history['acc'],label='train_acc')
    plt.legend(loc='lower center')
    plt.plot(epoch_max, h.history['val_acc'][epoch_max],'*')
    plt.show()


if __name__ == "__main__":
    main()


