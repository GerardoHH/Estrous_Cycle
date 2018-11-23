import numpy as np
import matplotlib.pyplot as plt

#from keras.datasets import mnist    # Keras importa la base de datos MNIST
from keras.models import Sequential # Sequential sirve para declarar modelos neuronales
from keras.layers.core import Dense, Dropout, Activation # Funciones para declarar capas neuronales específicas 
from keras.utils import np_utils    # np_utils sirve para crear los vectores objetivo
from keras import optimizers

from Datasource import Datasource as ldts

from model.DendralNeuron import DendralNeuron

#Npoints = 50

#x = np.linspace(-1,1,Npoints)
#y1 = np.cos(4*x)
#y2 = np.cos(4*x)-1

#load espiral dataset

np.random.seed(123456789)

#espiral_path = "/home/robotica/workspace/Keras/Espiral/Dataset/espiral/class_2/espiral_1.mat"
#P, T, Ptest, Ttest = ldts.loadDataset_Espiral_2Class_N_Loops ( espiral_path  )

P, T, Ptest, Ttest = ldts.loadDataset_Separable()



half = int(P.shape[0]/2)
        
x1 = P[0: half, 0]
y1 = P[0: half, 1]  

x2 = P[half:half*2, 0]
y2 = P[half:half*2, 1]  
        

plt.scatter(x1,y1)
plt.scatter(x2,y2)

plt.show()

Xblue = np.vstack((x1,y1)).T
Xred = np.vstack((x2,y2)).T

X = np.vstack((Xblue,Xred))


yblue = np.ones(Xblue.shape[0])
yred = np.zeros(Xred.shape[0])

Y = np.hstack((yblue,yred))


tlp = Sequential()

tlp.add(Dense(2, use_bias=True, activation='tanh', input_shape=(2,)))
#tlp.add( DendralNeuron(1, activation= 'tanh', input_shape=(2,)))

tlp.add(Dense(1, use_bias=True, activation='sigmoid'))

model = tlp
adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

h = model.fit(X, Y, batch_size=1024, epochs=100, verbose=1, validation_split=0.0, shuffle=True)


hlayer = Sequential()
hlayer.add(Dense(2, use_bias=True, activation='tanh', input_shape=(2,)))
#hlayer .add( DendralNeuron(1, activation= 'tanh', input_shape=(2,)))


xx = np.linspace(-1,1,100)
w2 = model.layers[1].get_weights()[0]
b2 = model.layers[1].get_weights()[1]
plt.plot(xx,(-w2[0][0]*xx - b2[0])/w2[1][0],'k')

y = hlayer.predict(X)
plt.plot(y[0:Xblue.shape[0],0],y[0:Xblue.shape[0],1],'b',y[Xblue.shape[0]+1:,0],y[Xblue.shape[0]+1:,1],'r')
plt.show()

def plot_decision_boundary(pred_func, X, Y, npts = 50):
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()
    dx, dy = (xmax - xmin)*0.1, (ymax - ymin)*0.1
    xx, yy = np.meshgrid(np.linspace(xmin-dx, xmax+dx, npts), np.linspace(ymin-dy, ymax+dy, npts))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, colors='k', levels=[0.48, 0.52])
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral, s=1)

plot_decision_boundary(lambda x: model.predict(x), X, Y, 100) 
plt.show()


# Entendiendo la transformación de la capa oculta

no = 50
x = np.linspace(-1,1,no)
y = np.linspace(-2,1,no)


for i in range(x.shape[0]):
    xo = x
    yo = y[i]*np.ones(no)
    yh = hlayer.predict(np.vstack((xo,yo)).T)
    
    plt.figure(1)
    plt.plot(xo,yo,'r') 
    plt.figure(2)
    plt.plot(yh[:,0],yh[:,1],'r')
    
for j in range(y.shape[0]):
    xo = x[j]*np.ones(no)
    yo = y
    yh = hlayer.predict(np.vstack((xo,yo)).T)
    
    plt.figure(1)
    plt.plot(xo,yo,'b') 
    plt.figure(2)
    plt.plot(yh[:,0],yh[:,1],'b') 
    
plt.figure(1)
plt.title('Espacio de entrada de la capa oculta')
plt.figure(2)
plt.title('Espacio de salida de la capa oculta')
plt.show()

print("Done ")
