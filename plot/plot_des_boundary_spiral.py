'''
Created on 25/09/2017

@author: robotica
'''
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
#from neural_architectures import *
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_pdf import PdfPages

from keras.models import Sequential
from keras.layers import Dense

from scipy.io import loadmat
import scipy.io 


from keras.utils import np_utils




def plot_decision_boundary_matlab(model, P, Y, npts = 50):
    
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    dx, dy = (xmax - xmin)*0.1, (ymax - ymin)*0.1
    
    xmin = -45
    xmax = 45
    ymin = -45
    ymax = 45
    
    xx, yy = np.meshgrid(np.linspace(xmin-dx, xmax+dx, Y.shape[0]), np.linspace(ymin-dy, ymax+dy, Y.shape[0]))
    z = model.predict( np.c_[xx.ravel(), yy.ravel()])
    #Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    plt.contour(xx, yy, z, colors='k', levels=[0.50, 0.51]) 
    #Original
    #plt.scatter(P[:, 0], P[:, 1], c=T, cmap=plt.cm.Spectral, s=1)
    half = int(P.shape[0]/2)
    plt.scatter(P[0: half, 0], P[0: half, 1]  , cmap=plt.cm.Spectral, s=1)
    plt.scatter(P[half:half*2, 0], P[half:half*2, 1]  , cmap=plt.cm.Spectral, s=1)
    
    plt.grid( axis='both')
    plt.show()
    

def model00():
   model = Sequential()
   model.add(Dense(200, activation='relu', bias=True,input_shape=(2,), init='normal'))
   model.add(Dense(100, activation='relu', bias=True, init='normal'))
   model.add(Dense(60, activation='relu', bias=True, init='normal'))
   model.add(Dense(40, activation='relu', bias=True, init='normal'))
   model.add(Dense(20, activation='relu', bias=True, init='normal'))
   model.add(Dense(10, activation='relu', bias=True, init='normal'))
   model.add(Dense(1, activation='sigmoid'))
   return model

def load_evaluated_mesh( mesh_path ):
    global Y, XX, YY, error
    
    dict = loadmat( mesh_path )
    
    Y  = np.asanyarray(dict['Y'])
    XX = np.asanyarray(dict['X1'])
    YY = np.asanyarray(dict['X2'])
    
    error = dict['Etest']
    
    Y[np.where(Y == 2)] = 0
    print("Mesh Loaded ....")

def load_Dataset ( dataset_path ):
    global P,T, Ptest, Ttest, nb_classes, input_dim
    
    dict = loadmat( dataset_path )
    
    t_p = dict['P']
    t_t = dict['T']
    t_ptest = dict ['Ptest']
    t_ttest = dict ['Ttest']
    
    
    t_P = np.array( t_p, dtype = np.float32)
    P = np.array( t_p, dtype = np.float32)
    
    t_T = np.array( t_t, dtype = np.int )
    T = np.array( t_t, dtype = np.int )
    
    t_Ptest = np.array( t_ptest, dtype = np.float32 )
    Ptest = np.array( t_ptest, dtype = np.float32 )
    
    t_Ttest = np.array( t_ttest, dtype = np.int )
    Ttest = np.array( t_ttest, dtype = np.int )
    
    del t_p
    del t_t
    del t_ptest
    del t_ttest
    
    P = np.zeros( [t_P.shape[1], t_P.shape[0]], np.float32 )
    T = np.zeros( [t_T.shape[1], t_T.shape[0]] , np.int )
    
    Ptest = np.zeros( [t_Ptest.shape[1],t_Ptest.shape[0]], np.float32 )
    Ttest = np.zeros( [t_Ttest.shape[1],t_Ttest.shape[0]], np.int )
    
    for idx in range( P.shape[0] ):
        P[idx ] = t_P[ :, idx ] 
        T[idx] = t_T[:, idx ]

    for idx in range ( Ptest.shape[0] ):
        Ptest[ idx ] = t_Ptest[ :, idx ]
        Ttest[ idx ] = t_Ttest[ :, idx ]
        
    del t_P
    del t_T
    del t_Ptest
    del t_Ttest
    
    input_dim = P.shape[1]
        
    T = T -1          # ESTO ES POR EL CATEGORICAL 
    Ttest = Ttest -1  # ESTO ES POR EL CATEGORICAL
    
    T = np_utils.to_categorical( T , nb_classes)
    Ttest = np_utils.to_categorical(Ttest, nb_classes)
    
    #print("\n\n ")
    print("\t Dataset Loaded ")
    print("\n\t Training: ")
    print("\t\t ---> P: " + str ( P.shape) )  
    print("\t\t ---> T: " + str ( T.shape) )
    
    print("\n\t Testing ")
    print("\t\t ---> Ptest: " + str( Ptest.shape ))
    print("\t\t ---> Ttest: " + str( Ttest.shape ))


def plot_decision_boundary_2_class(P, model, batch_size, h = 0.05, half_dataset = False, path_save = None, expand = 0.0):
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    
    xmin = xmin + xmin*expand
    xmax = xmax + xmax*expand
    
    ymin = ymin + ymin*expand
    ymax = ymax + ymax*expand
    
    
    xmin = -55
    xmax = 55
    
    ymin = -55
    ymax = 55
    #dx, dy = (xmax - xmin)*0.1, (ymax - ymin)*0.1
    
     #create mesh 
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))
    
    z = model.predict(np.c_[xx.ravel(), yy.ravel()] , batch_size ) # default batch size 32

    #Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z_t = z.reshape(xx.shape)
    
    
    plt.contour(xx, yy, z_t, colors='k', levels=[0.50, 0.51, 0.52])  #, 
    #plt.contourf(xx, yy, z_t, cmap=plt.cm.Paired, alpha=0.8)  #,
    #Original
    
    if ( half_dataset ):
        half = int(P.shape[0]/2)
        plt.scatter(P[0: half, 0], P[0: half, 1]  , cmap=plt.cm.Spectral, s=1)
        plt.scatter(P[half:half*2, 0], P[half:half*2, 1]  , cmap=plt.cm.Spectral, s=1)
    else:
        plt.scatter(P[:, 0], P[:, 1], cmap=plt.cm.Spectral, s=1)
    
    plt.grid(axis= 'both')
    
    if (path_save == None):
        plt.show()
    else:
        plt.savefig(path_save + 'desc_boundary.pdf', format='pdf')
        plt.close()
        
def main():
    global P,T, Ptest, Ttest, nb_classes, input_dim, nb_classes, Y, XX, YY, error
    
    Build = False
    
    nb_classes = 2
    load_Dataset('/home/robotica/workspace/Keras/Espiral/Dataset/espiral/class_2/espiral_5.mat')
    load_evaluated_mesh('/home/robotica/workspace/Keras/Espiral/Dataset/espiral/class_2/Decision_boundary/espiral_5.mat')
    path_save = '/home/robotica/workspace/Hybrid_MNN/Figs/hybrid/spiral/2_5/'
    pr_expand = 0.5
    model = model00()

    # Extract data from matlab file
    model.load_weights("/home/robotica/workspace/Keras/Espiral/Test/bestmodel_espiral_5vueltas_2clases.h5") 
     
    plot_decision_boundary_2_class(P, model, batch_size = 1024, h = 0.05, half_dataset = True, path_save=path_save, expand = pr_expand)
    
    
    
if __name__ == "__main__":
    main()
