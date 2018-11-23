'''
Created on Oct 9, 2018

@author: gerardo
'''

from keras import backend as K
from keras.models import Model
from keras import models

import matplotlib.pyplot as plt

import numpy as np

import cv2

def show_activations( model, X_val, y_val, show_plot = True ):
    
    print(" Visualiyzing  layers")
        
    X_val = np.asanyarray( X_val )
    y_val = np.asanyarray( y_val )
        
   
    layer_outputs_0 = []
    layer_outputs_1 = []
        
    for layer_idx in model.layers[:8] :                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        output_layer_idx_0 = layer_idx.get_output_at(0)
        output_layer_idx_1 = layer_idx.get_output_at(1)
        
        layer_outputs_0.append( output_layer_idx_0 )
        layer_outputs_1.append( output_layer_idx_1 )
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        
    activation_model_0 = models.Model(inputs=model.input, outputs=layer_outputs_0)
    #activation_model_1 = models.Model(inputs=model.input, outputs=layer_outputs_1)

    
    ground_truth  = model.predict( X_val)
    ground_truth  = np.argmax( ground_truth, axis =1)

    ground_truth  = ground_truth + 1 
         
    activations_0 = activation_model_0.predict( X_val )
    #activations_1 = activation_model_1.predict( X_val )

    first_layer_activation_0 = activations_0[0]
    #first_layer_activation_1 = activations_1[0]

    class_1 = False
    class_2 = False
    class_3 = False
    class_4 = False
    
    for idx_image  in range(0, X_val.shape[0]):

        if ( ground_truth[idx_image] == y_val[idx_image]):
        
            '''
            if ( ground_truth[idx_image] == 1  and class_1 == True):
                continue
                    
            if ( ground_truth[idx_image] == 2  and class_2 == True):
                continue
                
            if ( ground_truth[idx_image] == 3  and class_3 == True):
                continue
                   
            if ( ground_truth[idx_image] == 4  and class_4 == True):
                continue
                
            if ( ground_truth[idx_image] == 1 ):
                class_1 = True
                 
            if ( ground_truth[idx_image] == 2 ):
                class_2 = True
                 
            if ( ground_truth[idx_image] == 3 ):
                class_3 = True
                 
            if ( ground_truth[idx_image] == 4 ):
                class_4 = True

            '''
            
            print("Figure OK predicted: " + str(idx_image)  +  " Class: " +  str(y_val[idx_image] ))


            preds = model.predict( np.expand_dims( X_val[idx_image] , axis =0)  )
            
            activations_0 = activation_model_0.predict( np.expand_dims( X_val[idx_image] , axis =0)  )

            first_layer_activation_0 = activations_0[0]            

            images_per_row = 8

            n_cols= first_layer_activation_0.shape[-1] // images_per_row

            size = first_layer_activation_0.shape[1]
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            
            if (show_plot ):
                
                for col in range(n_cols):
                    for row in range(images_per_row):
                        
                        channel_image = first_layer_activation_0[0,:, :, col * images_per_row + row]
                        
                        channel_image -= channel_image.mean()
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

                scale = 1. / size
                plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
                plt.title("Convolutional layer actiations")

                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                #plt.show()

            #Plot GRAD-CAM
            idx_pred  = np.argmax(preds[0])
            model_output    = model.output[:, idx_pred]
            last_conv_layer = model.get_layer('block4_conv3')
            
            grads = K.gradients(model_output, last_conv_layer.get_output_at(1))[0]
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            
            iterate = K.function([model.input], [pooled_grads, last_conv_layer.get_output_at(1)])
            
            pooled_grads_value, conv_layer_output_value = iterate([  np.expand_dims( X_val[idx_image] , axis =0)  ])
            
            for i in range(512):
                    conv_layer_output_value[0, :, :, i] *= pooled_grads_value[i]
            
            heatmap = np.mean(conv_layer_output_value, axis=-1)
            
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            #plt.matshow(heatmap[0])
            
            heatmap = cv2.resize(heatmap[0], (X_val[idx_image].shape[1], X_val[idx_image].shape[0]))

            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            superimposed_img = heatmap * 0.25 + X_val[idx_image]
            

            path_save_heatmap = ''
            path_save_image   = ''
            
            if( ground_truth[idx_image] == 1):
                path_save_heatmap = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C1/'+ str(idx_image)+ '_Class_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                path_save_image   = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C1/'+ str(idx_image)+ '_Class_Img_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                path_save_grid    = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C1/'+ str(idx_image)+ '_Class_Grid_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                
            if( ground_truth[idx_image] == 2):
                path_save_heatmap = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C2/'+ str(idx_image)+ '_Class_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                path_save_image   = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C2/'+ str(idx_image)+ '_Class_Img_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                path_save_grid    = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C2/'+ str(idx_image)+ '_Class_Grid_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                
            if( ground_truth[idx_image] == 3):
                path_save_heatmap = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C3/'+ str(idx_image)+ '_Class_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                path_save_image   = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C3/'+ str(idx_image)+ '_Class_Img_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                path_save_grid    = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C3/'+ str(idx_image)+ '_Class_Grid_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                
            if( ground_truth[idx_image] == 4):
                path_save_heatmap = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C4/'+ str(idx_image)+ '_Class_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                path_save_image   = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C4/'+ str(idx_image)+ '_Class_Img_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'
                path_save_grid    = '/home/gerardo/Documents/workspace/Hybrid_MNN/Figs/estrous/C4/'+ str(idx_image)+ '_Class_Grid_'+str(y_val[idx_image]) + '_Idx_' +str(idx_image)+'.pdf'

            fig = plt.imshow(superimposed_img, cmap='jet' )            
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)        
            plt.savefig( path_save_heatmap )

            fig = plt.imshow(X_val[idx_image], cmap='jet' )
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)            
            plt.savefig( path_save_image )

            fig = plt.imshow(display_grid, aspect='auto', cmap='viridis')

            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)            
            plt.savefig( path_save_grid )


    print( "Done ... " )

