'''
Created on 18/12/2017

@author: robotica
'''


from scipy import misc

import math
from random import shuffle

from keras.datasets import mnist
from keras.datasets import cifar10

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import cv2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

from matplotlib import pyplot as plt 

import numpy as np

from sklearn.model_selection import KFold

from utils import estrous_features_extraction as efe
 

def write_to_filesystem(X_train, X_test, X_val, y_train, y_test, y_val):

    train_path = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/cross_validation/train/'
    test_path  = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/cross_validation/test/'
    val_path = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/cross_validation/validation/'
    
    for idx in range(len(X_train)):
        classe = y_train[idx]
        
        temp_path = train_path
        if ( classe == 1):
            temp_path = temp_path  + 'C1/'
        if ( classe == 2):
            temp_path = temp_path  + 'C2/'
        if ( classe == 3):
            temp_path = temp_path  + 'C3/'
        if ( classe == 4):
            temp_path = temp_path  + 'C4/'
        
        cv2.imwrite(temp_path+ str(idx) + ".png", X_train[idx])
          
          
    for idx in range(len(X_test)):
        classe = y_test[idx]
        
        temp_path = test_path
        if ( classe == 1):
            temp_path = temp_path  + 'C1/'
        if ( classe == 2):
            temp_path = temp_path  + 'C2/'
        if ( classe == 3):
            temp_path = temp_path  + 'C3/'
        if ( classe == 4):
            temp_path = temp_path  + 'C4/'
        
        cv2.imwrite(temp_path+ str(idx) + ".png", X_train[idx])
     
     
    for idx in range(len(X_val)):
        classe = y_val[idx]
        
        temp_path = val_path
        if ( classe == 1):
            temp_path = temp_path  + 'C1/'
        if ( classe == 2):
            temp_path = temp_path  + 'C2/'
        if ( classe == 3):
            temp_path = temp_path  + 'C3/'
        if ( classe == 4):
            temp_path = temp_path  + 'C4/'
        
        cv2.imwrite(temp_path+ str(idx) + ".png", X_train[idx])
     
     
    print(" Done exporting")

def load_K_Fold_dataset ( num_classes, to_one_hot, resize, k_fold ):
    
    import os

    base_path   = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/Data set/'
    output_path_train = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/cross_validation/train/'
    output_path_test = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/cross_validation/test/'
    output_path_validation = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/cross_validation/validation/'
    
    os.chdir(base_path)

    DIRECTORIOS=['C1','C2']

    images = []
    labels = []
    names  = []
    
    D = 0
    M = 0
    E = 0
    P = 0
    
    if ( num_classes == 4 ):
        for directorio in DIRECTORIOS:
            os.chdir( base_path + directorio )
            Imagenes_unsorted=os.listdir()
            Imagenes = sorted(Imagenes_unsorted)
                    
            for imagen in Imagenes:
                #print(imagen)
                img = cv2.imread(imagen,1)
                
                #pyplot.imshow( img )
                img = cv2.resize(img, resize)
                #resize( img, output_shape= (100,100,3), mode ='constant' )
                
                images.append(img)
                #plt.imshow(img)
                #plt.show()
                if directorio == DIRECTORIOS[0]: #directorio=='C1':
                    if imagen[0]=='D':
                        labels.append(1)
                        D += 1
                    else:
                        labels.append(2)
                        M += 1 
                else:
                    if imagen[0]=='E':
                        labels.append(3)
                        E += 1
                    else:
                        labels.append(4)
                        P += 1

                names.append(imagen)
    
        print("\nMouse Cells dataset loaded: ")
        print("\n\t\tTotal images  " + str(len(images)) + "  Total targets: " + str(len(labels)) )
        print("\n\t\t\tDiestro :  " + str(D))
        print("\n\t\t\tProestro:  " + str(P))
        print("\n\t\t\tMetaestro: " + str(M))
        print("\n\t\t\tEstro:     " + str(E))
        print("\n\t\tDataset done loading ... ")
    
        """
            D == Diestro    Etiqueta == 1
            P == Proestro   Etiqueta == 4
            M == Metaestro  Etiqueta == 2
            E == Estro      Etiqueta == 3
        """ 
    if ( num_classes == 2):
        """
            Class1 (Diestrus and Metestrus) 
            Class 2 (Estrus and Proestrus)
        """
        C1=0
        C2=0
        for directorio in DIRECTORIOS:
            os.chdir( base_path + directorio )
            
            Imagenes_unsorted=os.listdir()
            Imagenes = sorted(Imagenes_unsorted)
            
            for imagen in Imagenes:
                #print(imagen)
                img = cv2.imread(imagen,1)
                
                #pyplot.imshow( img )
                img = cv2.resize(img, resize)
                
                images.append(img)
                #plt.imshow(img)
                #plt.show()
                if directorio == DIRECTORIOS[0]: #directorio=='C1':
                    labels.append(1) 
                    C1+=1
                else:
                    labels.append(2)
                    C2+=1
                    
                names.append(imagen)
                    
        print("\nMouse Cells dataset loaded: ")
        print("\n\t\tTotal images  " + str(len(images)) + "  Total targets: " + str(len(labels)) )
        print("\n\t\t\tClass 1 (Diestrus and Metestrus) : {} ".format(C1))
        print("\n\t\t\tClass 2 (Estrus and Proestrus)   : {} ".format(C2))
        
        print("\n\t\tDataset done loading ... ")    
        

    #Split Cross Validation images
    print("\n\t\tSplitting  cross validation images")

    X_train, X_val, y_train, y_val = train_test_split( images, labels, test_size =0.15,  shuffle=True )
    
    X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size =0.20,  shuffle=True )    
      
    X= np.concatenate( [X_train ,  X_test ] )
    y= np.concatenate( [y_train , y_test  ] ) 
    
    
    del X_train
    del X_test
    del y_train
    del y_test   

    X_train = []
    y_train = []
    
    X_test = []
    y_test = []
    
    kf = KFold(n_splits= k_fold )

    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)

        X_train.append( X[train_index] ) 
        X_test.append (  X[test_index] )
        
        y_train.append( y[train_index] )
        y_test.append ( y[test_index]  ) 
        
    #Extract  105 features for  K_Folds
    for  idx in range(0, k_fold):
        f_train_idx,_,_ = efe.extract_features( X_train[idx]  )
        f_test_idx ,_,_ = efe.extract_features( X_test[idx]   )

        X_train[idx] = np.transpose( f_train_idx )
        X_test[idx]  = np.transpose( f_test_idx  )
        
        if ( to_one_hot  == True ):
            y_train[idx] = to_categorical(y_train[idx]-1, num_classes)
            y_test [idx] = to_categorical(y_test [idx]-1, num_classes)


    X_val   = np.asanyarray(X_val)
    y_val   = np.asanyarray( y_val)    

    X_val,_,_ = efe.extract_features( X_val  ) 
    X_val = np.transpose( X_val)
    
    if ( to_one_hot == True ):
        y_val = to_categorical(y_val-1, num_classes)

    X_train = np.asanyarray(X_train)
    X_test  = np.asanyarray(X_test )
    
    
    y_train = np.asanyarray( y_train )
    y_test  = np.asanyarray( y_test  )
        
    
    print("Train Folds " + str(X_train.shape ))
    print("\t\t " + str(y_train.shape ))
    
    print("Test Folds "  + str(X_test.shape  ))
    print("\t\t " + str(y_test.shape ))


    return  X_train, X_test, X_val, y_train, y_test, y_val
    
    
    
    
def spilt_dataset_CrossValidation( num_classes, normalize, to_one_hot, resize  ):
    
    import os

    base_path   = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/Data set/'
    output_path_train = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/cross_validation/train/'
    output_path_test = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/cross_validation/test/'
    output_path_validation = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/cross_validation/validation/'
    
    os.chdir(base_path)

    DIRECTORIOS=['C1','C2']

    images = []
    labels = []
    names  = []
    
    D = 0
    M = 0
    E = 0
    P = 0
    
    if ( num_classes == 4 ):
        for directorio in DIRECTORIOS:
            os.chdir( base_path + directorio )
            Imagenes_unsorted=os.listdir()
            Imagenes = sorted(Imagenes_unsorted)
                    
            for imagen in Imagenes:
                #print(imagen)
                img = cv2.imread(imagen,1)
                
                #pyplot.imshow( img )
                img = cv2.resize(img, resize)
                #resize( img, output_shape= (100,100,3), mode ='constant' )
                
                images.append(img)
                #plt.imshow(img)
                #plt.show()
                if directorio == DIRECTORIOS[0]: #directorio=='C1':
                    if imagen[0]=='D':
                        labels.append(1)
                        D += 1
                    else:
                        labels.append(2)
                        M += 1 
                else:
                    if imagen[0]=='E':
                        labels.append(3)
                        E += 1
                    else:
                        labels.append(4)
                        P += 1

                names.append(imagen)
    
        print("\nMouse Cells dataset loaded: ")
        print("\n\t\tTotal images  " + str(len(images)) + "  Total targets: " + str(len(labels)) )
        print("\n\t\t\tDiestro :  " + str(D))
        print("\n\t\t\tProestro:  " + str(P))
        print("\n\t\t\tMetaestro: " + str(M))
        print("\n\t\t\tEstro:     " + str(E))
        print("\n\t\tDataset done loading ... ")
    
        """
            D == Diestro    Etiqueta == 1
            P == Proestro   Etiqueta == 4
            M == Metaestro  Etiqueta == 2
            E == Estro      Etiqueta == 3
        """ 
    if ( num_classes == 2):
        """
            Class1 (Diestrus and Metestrus) 
            Class 2 (Estrus and Proestrus)
        """
        C1=0
        C2=0
        for directorio in DIRECTORIOS:
            os.chdir( base_path + directorio )
            
            Imagenes_unsorted=os.listdir()
            Imagenes = sorted(Imagenes_unsorted)
            
            for imagen in Imagenes:
                #print(imagen)
                img = cv2.imread(imagen,1)
                
                #pyplot.imshow( img )
                img = cv2.resize(img, resize)
                
                images.append(img)
                #plt.imshow(img)
                #plt.show()
                if directorio == DIRECTORIOS[0]: #directorio=='C1':
                    labels.append(1) 
                    C1+=1
                else:
                    labels.append(2)
                    C2+=1
                    
                names.append(imagen)
                    
        print("\nMouse Cells dataset loaded: ")
        print("\n\t\tTotal images  " + str(len(images)) + "  Total targets: " + str(len(labels)) )
        print("\n\t\t\tClass 1 (Diestrus and Metestrus) : {} ".format(C1))
        print("\n\t\t\tClass 2 (Estrus and Proestrus)   : {} ".format(C2))
        
        print("\n\t\tDataset done loading ... ")    
        

    #Split Cross Validation images
    print("\n\t\tSplitting  cross validation images")
    
    X_train, X_val, y_train, y_val = train_test_split( images, labels, test_size =0.15,  shuffle=True )
    
    X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size =0.20,  shuffle=True )
    
    
    y_train = np.asanyarray( y_train, dtype = np.int )
    y_test = np.asanyarray( y_test, dtype = np.int )
    y_val = np.asanyarray( y_val, dtype = np.int )
    
    
    if ( normalize == True ):
        X_train = np.asanyarray( X_train, dtype = np.float32 )
        X_test = np.asanyarray( X_test, dtype = np.float32 )
        X_val = np.asanyarray( X_val, dtype = np.float32 )
    
        X_train =  X_train / 255.0
        X_test  =  X_test  / 255.0
        X_val   =  X_val   / 255.0
    
    
    if ( to_one_hot == True ):
        # to one hot encoding
        y_train = y_train - 1
        y_test  = y_test  - 1
        y_val   = y_val   - 1
        
        y_train = to_categorical(y_train, num_classes)
        y_test  = to_categorical(y_test , num_classes)
        y_val   = to_categorical(y_val  , num_classes)


    if ( normalize ):    
        print("\n\t\tCross Val:")
        print("\n\t\t\t Train: " + str(X_train.shape))
        print("\n\t\t\t Test: " + str(X_test.shape))
        print("\n\t\t\t Val: " + str(X_val.shape))
    
    return  X_train, X_test, X_val, y_train, y_test, y_val

    
def augment_image( img_path, output_path, augment_num, prefix ):
    
    img = load_img(img_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    
    datagen = ImageDataGenerator( rotation_range=180,
                                  width_shift_range=0.05,
                                  height_shift_range=0.05,
                                  shear_range=0.05,
                                  zoom_range=0.05,
                                  horizontal_flip=True,
                                  vertical_flip = True,
                                  fill_mode='nearest' )    
    
    
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir= output_path,
                              save_prefix= prefix, save_format='jpg'):
        i += 1
        if i > augment_num:
            break
    
def augment_Mouse_cell_Dataset( num_images ):
    
    import os

    base_path   = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/Data set/'
    output_path_C1 = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/augmented_Data_set/C1/'
    output_path_C2 = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/augmented_Data_set/C2/'
    
    os.chdir(base_path)

    DIRECTORIOS=['C1','C2']


    for directorio in DIRECTORIOS:
        os.chdir( base_path + directorio )
        
        Imagenes=os.listdir()
            
        for imagen in Imagenes:

            if directorio=='C1':
                if imagen[0]=='D':
                    augment_image( base_path+'C1/'+imagen, output_path_C1, num_images, 'D__' )
                else:
                    augment_image( base_path+'C1/'+imagen, output_path_C1, num_images, 'M__' )
                
            else:
                if imagen[0]=='E':
                    augment_image( base_path+'C2/'+imagen, output_path_C2, num_images, 'E__' )
                #    E += 1
                else:
                    augment_image( base_path+'C2/'+imagen, output_path_C2, num_images, 'P__' )
                #        P += 1 

    print("\n\t Done augmenting ...")
       
def load_Mouse_cells( nFolds, num_classes, augmented_dataset = False , load_Folds = False):

    import os

    if ( augmented_dataset == True ):
        base_path = '/home/gerardo/Documents/workspace/Hybrid_MNN/utils/estrous_feature_extraction/'
        DIRECTORIOS=['C1_aug','C2_aug']
    else:
        base_path = '/home/gerardo/Documents/Doctorado/bases de datos/mouse_cells/Data set/'
        DIRECTORIOS=['C1','C2']
        
    os.chdir(base_path)

    

    labels = []
    images = []            
    
    D = 0
    P = 0
    M = 0
    E = 0
    
    if ( num_classes == 4 ):
        for directorio in DIRECTORIOS:
            os.chdir( base_path + directorio )
            Imagenes=os.listdir()
            
            for imagen in Imagenes:
                #print(imagen)
                img = imread(imagen)
                
                #pyplot.imshow( img )
                
                img = resize( img, output_shape= (100,100,3), mode ='constant' )
                images.append(img)
                #plt.imshow(img)
                #plt.show()
                if directorio == DIRECTORIOS[0]: #directorio=='C1':
                    if imagen[0]=='D':
                        labels.append(1)
                        D += 1 
                    else:
                        labels.append(2)
                        M += 1 
                else:
                    if imagen[0]=='E':
                        labels.append(3)
                        E += 1
                    else:
                        labels.append(4)
                        P += 1 
    
        print("\n Mouse Cells dataset loaded: ")
        print("\n\t\t  Total images  " + str(len(images)) + "  Total targets: " + str(len(labels)) )
        print("\n\t\t\t Diestro :  " + str(D))
        print("\n\t\t\t Proestro:  " + str(P))
        print("\n\t\t\t Metaestro: " + str(M))
        print("\n\t\t\t Estro:     " + str(E))
        print("\n Dataset done loading ... ")
    
        """
            D == Diestro    Etiqueta == 1
            P == Proestro   Etiqueta == 4
            M == Metaestro  Etiqueta == 2
            E == Estro      Etiqueta == 3
        """ 
    if ( num_classes == 2):
        """
            Class1 (Diestrus and Metestrus) 
            Class 2 (Estrus and Proestrus)
        """
        C1=0
        C2=0
        for directorio in DIRECTORIOS:
            os.chdir( base_path + directorio )
            Imagenes=os.listdir()
            
            for imagen in Imagenes:
                #print(imagen)
                img = imread(imagen)
                
                #pyplot.imshow( img )
                
                img = resize( img, output_shape= (100,100,3), mode ='constant' )
                images.append(img)
                #plt.imshow(img)
                #plt.show()
                if directorio == DIRECTORIOS[0]: #directorio=='C1':
                    labels.append(1) 
                    C1+=1
                else:
                    labels.append(2)
                    C2+=1
                    
        print("\n Mouse Cells dataset loaded: ")
        print("\n\t\t  Total images  " + str(len(images)) + "  Total targets: " + str(len(labels)) )
        print("\n\t\t\t Class 1 (Diestrus and Metestrus) : {} ".format(C1))
        print("\n\t\t\t Class 2 (Estrus and Proestrus)   : {} ".format(C2))
        
        print("\n Dataset done loading ... ")


    images = np.asanyarray(images, dtype= np.float32 )
    labels = np.asanyarray(labels, dtype= np.float32 )


    image_train_folds = []
    label_train_folds = []
    
    image_test_folds = []
    label_test_folds = []
    
    
    it = 0
    
    #labels = np.reshape(labels,[labels.shape[1],]) # Necesario cambiar dimension de T para StratifiedFold
     
    skf = StratifiedKFold(n_splits=nFolds, shuffle=True)
    
    hist_idx_test = []
    it = 0
    for it, (train_index, test_index) in enumerate(skf.split(images , labels)):
       
        X_train, X_test = images[train_index], images[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        image_train_folds.append( X_train )
        image_test_folds.append( X_test )
        
        y_train = y_train -1
        y_test  = y_test -1
        
        y_train = to_categorical(y_train, num_classes)
        y_test  = to_categorical(y_test , num_classes)
        
        
        label_train_folds.append( y_train )
        label_test_folds.append( y_test )
        
        print("Fold : ", it ," Train:", str( X_train.shape ), " Test:", str( X_test.shape) )
        
        hist_idx_test.append( test_index )
        it += 1

    #Shape 10 --> Folds
           ## --> Samples
           ## --> W_image
           ## --> H_image
           ## --> Chanels

    return   image_train_folds, label_train_folds, image_test_folds, label_test_folds, num_classes

def load_CIFAR10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    
    print("\n\tCIFAR-10 loaded")
    print('\n\t\tx_train shape: ', x_train.shape)
    print("\t\t\t", x_train.shape[0], 'train samples')
    print("\t\t\t", x_test.shape[0], 'test samples')
    
    img_rows = 32 
    img_cols = 32
    
    if K.image_data_format() == 'channels_first':
         x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
         x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
         input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)
            
    num_classes = 10
    
    
    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test , num_classes)

    return x_train, y_train, x_test, y_test, input_shape, num_classes

def load_MNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    print("\n\tMNIST loaded")
    print('\n\t\tx_train shape: ', x_train.shape)
    print("\t\t\t", x_train.shape[0], 'train samples')
    print("\t\t\t", x_test.shape[0], 'test samples')
    
    img_rows =28 
    img_cols = 28
    
    if K.image_data_format() == 'channels_first':
         x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
         x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
         input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
            
    num_classes = 10
    
    
    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test , num_classes)

    return x_train, y_train, x_test, y_test, input_shape, num_classes

def init_Cat_Dog_image_generator( batch_size ):
      
    X_train_idx = np.linspace(1, 12497, 12497, dtype= np.int32)

    # Partir  X_train en  train  and validation indices 
    shuffle( X_train_idx )
    
    X_train = X_train_idx[ 0 : int(len(X_train_idx)* 0.80) ]
    X_test  = X_train_idx[ int(len(X_train_idx)* 0.80) + 1: len(X_train_idx) ]
    
    train_image_generator = train_Cat_Dog_generator( X_train, batch_size)
    test_image_generator = train_Cat_Dog_generator( X_test, batch_size)
    
    return train_image_generator, test_image_generator , len(X_train)
    
def train_Cat_Dog_generator( samples, batch_size ):
    cat_path = '/home/gerardo/Documents/workspace/Keras/Espiral/Dataset/Cat_And_Dog/resized/Cat/'
    dog_path = '/home/gerardo/Documents/workspace/Keras/Espiral/Dataset/Cat_And_Dog/resized/Dog/'
    
    num_samples = len(samples)
   
    while True: #Loops forever so the generator never stops 
    
        for offset in range (0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            labels = []
            
            for batch_sample in batch_samples:
                #AQUI LEER LAS IMAGENES
                
                cat_img = cv2.imread( cat_path + str(batch_sample) + '.png' )
                dog_img = cv2.imread( dog_path + str(batch_sample) + '.png' )
                
                cat_img = np.asanyarray(cat_img, dtype = np.float32 )
                dog_img = np.asanyarray(dog_img, dtype = np.float32 )
                
                cat_img /= 255.0
                dog_img /= 255.0
                
                images.append(cat_img)
                images.append(dog_img)
                
                labels.append(0)
                labels.append(1)
                
            # trim image to only see section with road
            labels =  to_categorical( labels, num_classes=2)
            X_train = np.array(images)
            y_train = np.array(labels)
            yield X_train, y_train
            
def idx_gen_Cat_Dog():
    
    samples_idx = np.linspace(1, 100, num=100, dtype=np.int32)
    target_idx = np.zeros( len(samples_idx) )
    batch_size = 10
    
    X_train_idx, X_test_idx, _, _= train_test_split(  samples_idx, target_idx, test_size=0.33, random_state=42)
    
    return 

def load_Cat_Dog_Dataset( load_dataset = True ):
    
    cat_images = '/home/robotica/workspace/Keras/Espiral/Dataset/Cat_And_Dog/PetImages/Cat/'
    dog_images = '/home/robotica/workspace/Keras/Espiral/Dataset/Cat_And_Dog/PetImages/Dog/'
    
    dest_dir_cat = '/home/robotica/workspace/Keras/Espiral/Dataset/Cat_And_Dog/resized/Cat/'
    dest_dir_dog = '/home/robotica/workspace/Keras/Espiral/Dataset/Cat_And_Dog/resized/Dog/'
    
    # 12501 images in both
    cats = []
    dogs = []
    
    if ( load_dataset ):
        print("--> Loading  Cats and Dogs datasets ")
            
        for i in range (1, 12497):
            try:
                cat = misc.imread( dest_dir_cat + str(i) +'.png' )
                dog = misc.imread( dest_dir_dog + str(i) +'.png' )
                
                cat = np.asanyarray(cat, dtype = np.float32 )
                dog = np.asanyarray(dog, dtype = np.float32 )
                
                cat /= 255.0
                dog /= 255.0

                cats.append( cat )
                dogs.append( dog )

            except:
                print( "Image failed  " + str ( i ))
                
        print("\t\t  Cats --> " + str(len(cats)) )
        print("\t\t  Dogs --> " + str(len(dogs)) )
        
        P = cats + dogs
    
        zero_cats = np.zeros( len(cats) )
        ones_dogs = np.ones(  len(dogs) ) 
        T = np.concatenate( (zero_cats, ones_dogs) )
        
        del cats
        del dogs
        
        P, Ptest, T, Ttest = train_test_split( P, T, test_size=0.33, random_state=42)
        
        print("Dataset Loaded ")
        print("\t Train --> P: " +str(P.shape) + "\t T: " +str(T.shape))
        print("\t Test  --> Ptest: " +str(Ptest.shape) + "\t Ttest: " +str(Ttest.shape))
        
    else:
        for i in range(1 , 12501):
            #print( cat_images + str(i) + '.jpg' )
            #print( dog_images + str(i) + '.jpg' )
            
            try:
                cat_img = Image.open(cat_images + str(i) + '.jpg' )
                dog_img = Image.open(dog_images + str(i) + '.jpg' )
    
                w_resize = 224
                h_resize = 224
            
                cat_img = cat_img.resize((w_resize,h_resize), Image.ANTIALIAS)
                dog_img = dog_img.resize((w_resize,h_resize), Image.ANTIALIAS)
            
                cat_img.save( dest_dir_cat + str(i) + '.png' )
                dog_img.save( dest_dir_dog + str(i) + '.png' )
            
                if ( math.fmod(i, 100)  == 0 ):
                    print( "processing " + str ( i ))
            except:
                    print( "Image failed  " + str ( i ))
     
               
    return  P, T, Ptest, Ttest
        #cat_img = misc.imread( cat_images + str(i) + '.jpg' )
        #dog_img = misc.imread( dog_images + str(i) + '.jpg' )
        
        #plt.imshow(cat_img)
        #plt.show()
        
        #plt.imshow(dog_img)
        #plt.show()
        
        #escalar en tamaño y normalizar  a [0-1]
        
