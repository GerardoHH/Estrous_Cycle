
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.utils import np_utils
from keras import models, layers


from Datasource import Image_Datasource as img_dts
from training import BuildModel as bm

from model import Utils_Model as um

from plot import plot_utils

from matplotlib import cm


from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import sklearn.grid_search as gs


from utils import estrous_features_extraction as efe

from visualization import visualize_Conv_Layer 


import cv2

import random as rn
import numpy as np

from keras import backend as K

import os

os.environ['PYTHONHASHSEED'] = '0'


def build_train_VGG_16( num_classes, LR, batch_size, nb_epochs, nb_verbose, steps_per_epoch, resize_shape, regularizar, test_model = False):
    
    X_train, X_test, X_val, y_train, y_test, y_val = img_dts.spilt_dataset_CrossValidation( num_classes, True, True, resize_shape  )
    

    train_generator = ImageDataGenerator( rotation_range=180,
                                  horizontal_flip=True,
                                  vertical_flip = True,
                                  fill_mode='nearest')    

    train_generator.fit(X_train)
    
    test_generator = ImageDataGenerator( )

    if ( test_model == False):
    
        #model =  um.build_vgg_16( num_classes, LR, True)
     
        model = um.build_custom_vgg_16 ( num_classes, LR, True)                  
    
        hist = um.train_VGG_16_Datagen(model, train_generator,  test_generator , batch_size, nb_epochs, X_train, y_train, X_test, y_test, nb_verbose, steps_per_epoch)
   

        idx_max_val   = np.argmax( hist.history['val_acc'] )
        
        print("\n\t---->Best Val Acc idx: " +str( idx_max_val ) + " Val_Acc_Value: " +str( hist.history['val_acc'][idx_max_val]) )
    
        print("\n Testing model:  ")
        
        score = model.evaluate(X_val, y_val, batch_size, verbose=False )
    
        print("\n\tTesting \t Test_Acc: " +str(score[1]) )
    
        #plot_utils.plot_train_No_loss( hist, path_save= None )
    
        output = model.predict(X_val, batch_size, verbose= False)
    
        #print("\n\t\t Predicted labels: " + str(output))
    
    
    if ( test_model == True):

        print("\n\t\t\tLoagind model ...")

        model_loded = um.build_custom_vgg_16 ( num_classes, LR, True)
                    
        model_loded.load_weights('/home/gerardo/Documents/workspace/Hybrid_MNN/trained_models/CNN-Custom_VGG16[0_-4]_4_classes_F1_0.82_Acc_0.80.h5')
        
        print("\n\t\t\tModel loaded ...")
        
        output = model_loded.predict(X_val, batch_size, verbose= False)

        print("\n\t\t\tPredicting ...")
        
        for idx in range( output.shape[0]):
            c_idx = np.argmax( output[idx])
            output[idx] = 0
            output[idx][c_idx ] = 1
        
        macro_f1_score = f1_score(y_val, output, average ='macro')
        micro_f1_score = f1_score(y_val, output, average ='micro')
    
        print("\n\t\t\t Saved Model: ")
        print("\n\t\t\t F1 micro score : " + str(micro_f1_score))
        print("\n\t\t\t F1 macro score : " + str(macro_f1_score))

        
    for idx in range( output.shape[0]):
        c_idx = np.argmax( output[idx])
        output[idx] = 0
        output[idx][c_idx ] = 1
        
        
    macro_f1_score = f1_score(y_val, output, average='macro')
    micro_f1_score = f1_score(y_val, output, average ='micro')
    
    print("\n\t\t F1 micro score : " + str(micro_f1_score))
    print("\n\t\t F1 macro score : " + str(macro_f1_score))
    
    
    print("\n\VGG_16 Classification Done ... ")
    
    plot_utils.my_plot_train_loss( hist )
    
    save_model = input(" Save model (y/n) ? ")
    
    save_model= str(save_model)
    
    print(save_model)
    
    if ( save_model == 'y'):

        print("\n\t\t\tSaving model ...")
        model.save('/home/gerardo/Documents/workspace/Hybrid_MNN/trained_models/vgg_16.h5')  # creates a HDF5 file 'my_model.h5'

        print("\n\t\t\tLoagind model ...")
        model_loded = load_model('/home/gerardo/Documents/workspace/Hybrid_MNN/trained_models/vgg_16.h5')
        
        print("\n\t\t\tModel loaded ...")
        
        output = model_loded.predict(X_val, batch_size, verbose= False)

        print("\n\t\t\tPredicting ...")
        
        for idx in range( output.shape[0]):
            c_idx = np.argmax( output[idx])
            output[idx] = 0
            output[idx][c_idx ] = 1
        
        macro_f1_score = f1_score(y_val, output, average ='macro')
        micro_f1_score = f1_score(y_val, output, average ='micro')
    
        print("\n\t\t\t Saved Model: ")
        print("\n\t\t\t F1 micro score : " + str(micro_f1_score))
        print("\n\t\t\t F1 macro score : " + str(macro_f1_score))

def biuld_train_SVM_RBF_Cross_val( grid_search, low_lim, high_lim, num_classes, resize_shape, steps_per_epoch, folds = 10  ):
    
    #instantiate  the generator 

    X_train, X_test, X_val, y_train, y_test, y_val =  img_dts.load_K_Fold_dataset(num_classes, to_one_hot = False, resize = resize_shape, k_fold = folds )
    
    print("\n******** Training  SVM --> RBF ******")
    print("\n\tStarting  Grid Search ......... ")

    est = gs.GridSearchCV(SVC(),
                          {'C'    : np.logspace( low_lim , high_lim , grid_search),
                           'gamma': np.logspace( low_lim , high_lim , grid_search)});
   
    C = est.param_grid['C']
    gamma = est.param_grid['gamma']
    
    
    best_test = 0
    best_val  = 0
    best_it   = 0 
    best_C    = 0
    best_gamma = 0
    idx = 0
    
    for idx_C in range(0, len(C)):
        
        for idx_gamma in range (0, len(gamma)):
            
            print ("\tIt: "+ str(idx))
    
            acum_acc = []
            for idx_ds in range( 0, X_train.shape[0] ):
                X_train_idx = X_train[idx_ds]
                y_train_idx = y_train[idx_ds]

                local_model = SVC(kernel='rbf', C=C[idx_C], gamma= gamma[idx_gamma], random_state=1)

                local_model.fit( X_train_idx, np.ravel(y_train_idx) )

                y_predict = local_model.predict( X_test[idx_ds])
                score_val  = accuracy_score(y_test[idx_ds], y_predict)

                acum_acc.append( score_val )

            score_test = np.average( acum_acc )
            
            if score_test > best_test :
                    best_test = score_test
                    best_val  =score_val
                    best_it = idx 
                    best_C = C[idx_C]
                    best_gamma = gamma[idx_gamma]
    
            print ("\n\t\tVal Acc: " + str(score_val) + "\tTest: " +str(score_test) + "\tC: "+ str(C[idx_C]) + "\tGamma: "+ str(gamma[idx_gamma]) )
            print ("\n\t\t\tBest It: "+ str(best_it)+"\tBest Test Acc: " +str(best_test) )        
        
            del acum_acc
            del local_model
            
            idx = idx + 1

   
    print("\n\tBest Model: ")
    print("\n\t\t Idx: " + str(best_it))
    print("\n\t\t Best Test Acc: " + str(best_test))
    print("\n\t\t Best Val Acc: " + str(best_val))
    print("\n\t\t Gamma: " + str(best_gamma) + "\tC: "+ str(best_C))
    
    
    
    #best_gamma =  6.892612104349695e-07    
    #best_C = 9326033.46883218
    
    acum_acc     = [ ]
    acum_val_macro    = [ ]
    acum_val_micro    = [ ]

    for idx_ds in range( 0, X_train.shape[0] ):
        X_train_idx = X_train[idx_ds]
        y_train_idx = y_train[idx_ds]

        local_m = SVC(kernel='rbf', C=best_C, gamma= best_gamma, random_state=1)
        local_m.fit( X_train_idx, np.ravel(y_train_idx) )
        
        y_predict = local_m.predict( X_test[idx_ds])
        score_val  = accuracy_score(y_test[idx_ds], y_predict)
        
        acum_acc.append( score_val )
               
        y_predict = local_m.predict( X_val )
        score_test = accuracy_score(y_val, y_predict)
        
        macro_f1_score = f1_score(y_val, y_predict, average='macro')
        micro_f1_score = f1_score(y_val, y_predict, average ='micro')
        
        acum_val_macro.append( macro_f1_score )
        acum_val_micro.append( micro_f1_score )
    
    print("\n\t\t Best models SVM RBF: ")
    print("\n\t\t\tModel.C: " + str(best_C) + "\tModel.gamma: " +str(best_gamma) )
    print("\n\t\t\tF1 Micro: " + str(np.average( acum_val_micro )))
    print("\n\t\t\tF1 Macro: " + str(np.average( acum_val_macro )))
    
def build_train_Conv_MLP_LeNet_Model_Cross_Val( num_classes, LR, batch_size, nb_epochs, nb_verbose, steps_per_epoch, resize_shape, regularizar):
    

    X_train, X_test, X_val, y_train, y_test, y_val = img_dts.spilt_dataset_CrossValidation( num_classes, True, True , resize_shape )
    

    train_generator = ImageDataGenerator( rotation_range=180,
                                  horizontal_flip=True,
                                  vertical_flip = True,
                                  fill_mode='nearest')

    train_generator.fit(X_train)
    
    test_generator = ImageDataGenerator( )

    input_shape = X_train.shape[1 :]
    
    model =  um.build_LeNet(input_shape, num_classes, LR, regularizar )
                
    hist = um.train_Lenet_Datagen(model, train_generator, test_generator, batch_size, nb_epochs, X_train, y_train, X_test, y_test, nb_verbose, steps_per_epoch)
        
    idx_max_val   = np.argmax( hist.history['val_acc'] )
        
    print("\n\t---->Best Val Acc idx: " +str( idx_max_val ) + " Val_Acc_Value: " +str( hist.history['val_acc'][idx_max_val]) )
    
    print("\n Testing model:  ")
        
    score = model.evaluate(X_val, y_val, batch_size, verbose=False )
    
    print("\n\tTesting \t Test_Acc: " +str(score[1]) )
    
    #plot_utils.plot_train_No_loss( hist, path_save= None )
    
    output = model.predict(X_val, batch_size, verbose= False)
    
    #print("\n\t\t Predicted labels: " + str(output))
    
    
    for idx in range( output.shape[0]):
        c_idx = np.argmax( output[idx])
        output[idx] = 0
        output[idx][c_idx ] = 1
        
        
    macro_f1_score = f1_score(y_val, output, average='macro')
    micro_f1_score = f1_score(y_val, output, average ='micro')
    
    print("\n\t\t F1 micro score : " + str(micro_f1_score))
    print("\n\t\t F1 macro score : " + str(macro_f1_score))
    
    
    print("\n\tConvNet Classification Done ... ")
    
    plot_utils.my_plot_train_loss( hist )
    
    save_model = input(" Save model (y/n) ? ")
    
    save_model= str(save_model)
    
    print(save_model)
    
    if ( save_model == 'y'):
        from keras.models import load_model

        print("\n\t\t\tSaving model ...")
        model.save('/home/gerardo/Documents/workspace/Hybrid_MNN/trained_models/convNet.h5')  # creates a HDF5 file 'my_model.h5'

        print("\n\t\t\tLoagind model ...")
        model_loded = load_model('/home/gerardo/Documents/workspace/Hybrid_MNN/trained_models/convNet.h5')
        
        print("\n\t\t\tModel loaded ...")
        
        output = model_loded.predict(X_val, batch_size, verbose= False)

        print("\n\t\t\tPredicting ...")
        
        for idx in range( output.shape[0]):
            c_idx = np.argmax( output[idx])
            output[idx] = 0
            output[idx][c_idx ] = 1
        
        macro_f1_score = f1_score(y_val, output, average ='macro')
        micro_f1_score = f1_score(y_val, output, average ='micro')
    
        print("\n\t\t\t Saved Model: ")
        print("\n\t\t\t F1 micro score : " + str(micro_f1_score))
        print("\n\t\t\t F1 macro score : " + str(macro_f1_score))
           
def build_train_DNN_Model ( min_num_of_layers, max_num_of_layers, max_neruons_x_layer, num_of_trials, min_LR, max_LR, activation, num_classes, batch_size, nb_epochs, nb_verbose, resize_shape, start_from, to_end, steps_per_epoch, folds ):

    #instantiate  the generator 

    X_train, X_test, X_val, y_train, y_test, y_val =  img_dts.load_K_Fold_dataset(num_classes, to_one_hot = True, resize = resize_shape, k_fold = folds )
    
    print(" Starting DNN Classification ... ")
    
    #Generate  grid search
    dnn_models, LR_arr =  um.generate_hyperparam_grid( min_num_of_layers, max_num_of_layers, max_neruons_x_layer, num_of_trials, min_LR, max_LR )
     
     
    #dnn_models = [ [204, 104, 77]      ]
    #LR_arr =     [ 0.00970027842467605 ] 
     
    train_time_hist = []
    b_val_acc = -1
    b_train_acc = -1
    
    pr_expand = 2
    
    for idx_model in range(0 + start_from , len(dnn_models)):
        nb_neurons = dnn_models[ idx_model]
        lr    = LR_arr[ idx_model ]
        
        model = bm.build_MLP_DN ( nb_neurons, X_train.shape[2],  num_classes,  activation)
    
        original_weights = model.get_weights()
            
        print("Iteration: "+ str(idx_model))
        print("\t ---> Training  arqui: " + str(nb_neurons) + "\tLR: " + str(lr)  )
        
        
        acum_test = []
        acum_train = []
        for idx_ds in range( 0, folds):
            X_train_idx = X_train[idx_ds]
            y_train_idx = y_train[idx_ds]

            X_test_idx = X_test[idx_ds]
            y_test_idx = y_test[idx_ds]
            
            [hist, train_time] = bm.train_HybridModel( model, lr, X_train_idx, y_train_idx, X_test_idx, y_test_idx, batch_size, nb_epochs, nb_verbose )
                
            best =  np.argmax( hist.history['val_acc'] )    
            acum_test.append(  hist.history['val_acc'] [best] )
            
            best_train = np.argmax( hist.history['acc'] )
            acum_train.append( hist.history['acc'][best_train]  )
            
            model.set_weights( original_weights  )
            
        train_time_hist.append(train_time)
        
        
        #best =  np.argmax( hist.history['val_acc'] )
        val_test_it = np.average( acum_test )
        
        if ( b_val_acc < val_test_it  ):
            #b_hist = hist
            b_idx =  best
            b_lr  = lr
            b_batch_size = batch_size 
            b_nb_epoch = nb_epochs 
            b_model = model
            b_nb_neurons = nb_neurons
            b_train_time = train_time
            b_val_acc = val_test_it
            b_train_acc = np.average( acum_train )
            b_best = best

        print("\t\t ---> Best \tAcc Train: " + str( b_train_acc ) + 
              " Val Acc: " + str( b_val_acc )  + 
              " LR: " + str( b_lr ) + 
              " batch_size: " + str( b_batch_size ) + 
              " nb_epoch: " + str( b_nb_epoch) + 
              " model_params: " + str( b_model.count_params()) +
              " Architecture : " + str( b_nb_neurons ) + 
              " Time: "+  str(b_train_time) +
              " Overall Time: " + str( np.sum(train_time_hist)))
        
        del hist
        del train_time
        del model
        
    output = b_model.predict(X_val, batch_size, verbose= False)

    output_tmp = output
    print("\n\t\t\tPredicting ...")
        
    for idx in range( output.shape[0]):
        c_idx = np.argmax( output[idx])
        output[idx] = 0
        output[idx][c_idx ] = 1
        
    macro_f1_score = f1_score(y_val, output, average ='macro')
    micro_f1_score = f1_score(y_val, output, average ='micro')
    
    print("\n\t\t\t Saved Model: ")
    print("\n\t\t\t F1 micro score : " + str(micro_f1_score))
    print("\n\t\t\t F1 macro score : " + str(macro_f1_score))
        
        
    #################################################################################
    ####### PLOT RESULTS  AND  DESCISSION BOUNDARY
    
    plot_utils.plot_train_loss( b_hist)
    
    #if ( output_shape == 2 and input_dim == 2 ):
    #    plot_utils.plot_decision_boundary_2_class(P, b_model, batch_size, h = 0.05, half_dataset = True, path_save=path_save, expand = pr_expand)
    
    #if (output_shape == 3 and input_dim == 2):
    #    plot_utils.plot_decision_boundary_3_class(P, b_model, batch_size, h = 0.05, half_dataset = True, path_save=path_save, expand = pr_expand)
    
    #if ( input_dim == 3):
    #    plot_utils.plot_decision_boundary_2_class_3D(P, b_model, batch_size, h = 0.08, half_dataset = True, path_save=path_save, expand = pr_expand)
        #plot_utils.plot_decision_boundary_2_class_3D(P_ori, model, batch_size, h = 0.5, half_dataset = True, path_save=path_save, expand = pr_expand)
    
 
    print(" Done DNN Classification ... ")
    
def visualize_VGG16_modified_model(num_classes, resize_shape = (150,150), LR =0, plot = False):
    
    #Primero construir un modelo vacio
    
    model = um.build_custom_vgg_16(num_classes, LR, plot)
    
    model.load_weights('/home/gerardo/Documents/workspace/Hybrid_MNN/trained_models/CNN-Custom_VGG16[0_-4]_4_classes_F1_0.88_Acc_0.80.h5' )
    
    print("\n Done loading ... ")

    print("\n Loading Dataset ... ")
    
    X_train, X_test, X_val, y_train, y_test, y_val = img_dts.spilt_dataset_CrossValidation( num_classes, True, False, resize_shape  )
    
    print("\n Done loading Dataset ... ")
         
    output = model.predict(X_val, batch_size = 128, verbose= False)

    print("\n\t\t\tPredicting ...")
        
    for idx in range( output.shape[0]):
        c_idx = np.argmax( output[idx])
        output[idx] = 0
        output[idx][c_idx ] = 1
        
    y_val_numeric = y_val 
    
    y_val = y_val -1
    y_val = to_categorical(y_val, num_classes)
    macro_f1_score = f1_score(y_val, output, average ='macro')
    micro_f1_score = f1_score(y_val, output, average ='micro')
    
    print("\n\t\t\t Saved Model: ")
    print("\n\t\t\t F1 micro score : " + str(micro_f1_score))
    print("\n\t\t\t F1 macro score : " + str(macro_f1_score))
    
    
    visualize_Conv_Layer.show_activations( model, X_val, y_val_numeric, show_plot = True )

def main():
    
    ####################################   REPRODUCIBILIDAD #######################################################
    ####################################   REPRODUCIBILIDAD #######################################################
    np.random.seed(12345)
    
    rn.seed(12345)

    import tensorflow as tf
    tf.set_random_seed(12345)


    ####################################   LOAD PRETRAINED MODEL  #################################################
    ####################################   lOAD PRETRAINED MODEL  #################################################

    #num_classes = 4
    #visualize_VGG16_modified_model(num_classes = num_classes )


    #####################################################################################################################
    #######################  TRAINING TESTING   VGG-16 ########################################################
    #####################################################################################################################

    
    num_classes = 2
    LR =0.001
    batch_size = 128

    nb_epochs =  7

    nb_verbose = False

    steps_per_epoch = 2800 
    
    resize_shape = (150,150) 
    
    regularizar = True
    
    test_model = False
    
    build_train_VGG_16( num_classes, LR, batch_size, nb_epochs, nb_verbose, steps_per_epoch, resize_shape, regularizar, test_model )
    
     
    #####################################################################################################################
    #######################  TRAINING TESTING   SVM -RBF   ##############################################################
    #####################################################################################################################

    
    num_classes = 2
    grid_search = 10000
    low_lim = -10
    high_lim = 10
    resize_shape = (150, 150)
    steps_per_epoch = 5600
    
    folds = 10
    
    biuld_train_SVM_RBF_Cross_val( grid_search, low_lim, high_lim, num_classes, resize_shape, steps_per_epoch, folds )

    
    #####################################################################################################################
    #######################  TRAINING TESTING   CONVNETS   ##############################################################
    #####################################################################################################################
    
    
    #Convolutional Hyper-parameters
    LR = 0.001
    nb_epochs = 5
    batch_size = 512
    nb_verbose = True
    steps_per_epoch = 2800 
    
    regularizar = True  # dropout --->  False    True --->  L1_L2

    resize_shape = (150, 150)

    num_classes = 2

    img_dts.augment_Mouse_cell_Dataset( num_images = 10 )
    build_train_Conv_MLP_LeNet_Model_Cross_Val(  num_classes, LR, batch_size, nb_epochs, nb_verbose, steps_per_epoch, resize_shape, regularizar)   

    
    #####################################################################################################################
    #######################  TRAINING TESTING   DNN   ###################################################################
    #####################################################################################################################

    min_num_of_layers = 1
    max_num_of_layers = 5

    max_neruons_x_layer = 250

    num_of_trials = 10000

    min_LR = 0.0001
    max_LR = 0.1
    activation = "tanh"
    num_classes = 2
    batch_size = 2048

    nb_epochs  = 500
    nb_verbose = False

    resize_shape = (150,150) 

    start_from = 0

    steps_per_epoch = 5600

    to_end = num_of_trials

    k_fold = 10 

    build_train_DNN_Model ( min_num_of_layers, max_num_of_layers, max_neruons_x_layer, num_of_trials, min_LR, max_LR, activation, num_classes, batch_size, nb_epochs, nb_verbose, resize_shape, start_from, to_end, steps_per_epoch, folds = k_fold  )
    
    
    K.clear_session()
    print("Done ")


if __name__ == "__main__":
    main()