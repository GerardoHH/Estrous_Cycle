'''
Created on Nov 6, 2018

@author: gerardo
'''

import matplotlib.pyplot as plt
import numpy as np




over_all_overfitting_test =  [ 0.9838709831, 0.9646713,  0.919354856 , 0.9742857158, 0.9142857143 ]
ovel_all_overfitting_val  =  [ 0.9571428299, 0.9254566,  0.9857142568, 0.9971428573, 0.9314285714 ]

over_all_overfitting_test =  np.asanyarray( over_all_overfitting_test ) 
ovel_all_overfitting_val  =  np.asanyarray( ovel_all_overfitting_val  )

x_axes = np.abs(over_all_overfitting_test  - ovel_all_overfitting_val )


arg_min = np.argmin( x_axes )


plt.figure
plt.plot( x_axes[0], over_all_overfitting_test[0], 'bo' , label = 'VGG-16-M', markersize = 12)
plt.plot( x_axes[1], over_all_overfitting_test[1], 'go' , label = 'VGG-16'  , markersize = 12)
plt.plot( x_axes[2], over_all_overfitting_test[2], 'ro' , label = 'LeNet-5' , markersize = 12)
plt.plot( x_axes[3], over_all_overfitting_test[3], 'ko' , label = 'MLP'     , markersize = 12)
plt.plot( x_axes[4], over_all_overfitting_test[4], 'co' , label = 'SVM'     , markersize = 12)



plt.tick_params(axis='y', labelsize=25)

plt.xlabel('|| test - val || Accuracy (2 Estrous cycles)', fontsize=35 )
plt.ylabel(' Test Accuracy (%)', fontsize= 35)

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,  ncol=5, mode="expand", borderaxespad=0., prop={'size':30} )

plt.grid(True)
plt.show()


###################################################################################################
###################################################################################################


over_all_labels = [ ]

 
over_all_overfitting_test =  [ 0.8225806355, 0.7665987, 0.8064516187, 0.8142857254,  0.6571428571 ]
ovel_all_overfitting_val  =  [ 0.8000000119, 0.8842714, 0.8142856956, 0.9146031797,  0.7028571429 ]
   
over_all_overfitting_test =  np.asanyarray( over_all_overfitting_test ) 
ovel_all_overfitting_val  =  np.asanyarray( ovel_all_overfitting_val  )

x_axes = np.abs(over_all_overfitting_test  - ovel_all_overfitting_val )


arg_min = np.argmin( x_axes )


plt.figure
plt.plot( x_axes[0], over_all_overfitting_test[0], 'bo' , label = 'VGG-16-M', markersize = 12)
plt.plot( x_axes[1], over_all_overfitting_test[1], 'go' , label = 'VGG-16'  , markersize = 12)
plt.plot( x_axes[2], over_all_overfitting_test[2], 'ro' , label = 'LeNet-5' , markersize = 12)
plt.plot( x_axes[3], over_all_overfitting_test[3], 'ko' , label = 'MLP'     , markersize = 12)
plt.plot( x_axes[4], over_all_overfitting_test[4], 'co' , label = 'SVM'     , markersize = 12)



plt.tick_params(axis='y', labelsize=25)

plt.xlabel('|| test - val || Accuracy (4 Estrous cycles)', fontsize=35 )
plt.ylabel(' Test Accuracy (%)', fontsize= 35)

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,  ncol=5, mode="expand", borderaxespad=0., prop={'size':30} )

plt.grid(True)
plt.show()

####################################################################
####################################################################
CNN_Custom_VGG16_Val_acc =  [ 0.8000000119, 0.7714285851, 0.7857142687, 0.8000000119, 0.8000000119, 0.7714285851, 0.7571428418, 0.7857142687, 0.7857142687, 0.7714285851]
CNN_Custom_VGG16_Test_acc = [ 0.8225806355,  0.919354856, 0.8548387289, 0.8225806355, 0.8387096524, 0.8548387289, 0.8870967627, 0.8225806355, 0.8548387289, 0.8709677458 ]


CNN_Custom_VGG16_Val_acc  = np.asanyarray( CNN_Custom_VGG16_Val_acc )
CNN_Custom_VGG16_Test_acc = np.asanyarray( CNN_Custom_VGG16_Test_acc ) 

y_axes =  np.abs (  CNN_Custom_VGG16_Val_acc - CNN_Custom_VGG16_Test_acc )

arg_min = np.argmin( y_axes )


print(" Custom VGG_16 " )
print(" Min diference: " + str( y_axes[arg_min] ) )
print(" Val_acc     : " + str(CNN_Custom_VGG16_Val_acc [arg_min]) )
print(" Val_test    : " + str(CNN_Custom_VGG16_Test_acc[arg_min]) )

plt.figure
plt.plot( y_axes, 'bo' )
plt.plot( y_axes[arg_min], 'r*')
plt.grid(True)
plt.show()







'''

plt.bar(x_axis               , HNN_FE   , bar_width, alpha = opacity, color = 'blue'             , label = 'HNN-FE')
plt.bar(x_axis +   bar_width , KNN      , bar_width, alpha = opacity, color = 'royalblue'        , label = 'KNN')
plt.bar(x_axis + 2*bar_width , BAYES    , bar_width, alpha = opacity, color = 'r'                , label = 'N-Bayes')
plt.bar(x_axis + 3*bar_width , D_TREE   , bar_width, alpha = opacity, color = 'g'                , label = 'D-Tree')
plt.bar(x_axis + 4*bar_width , R_FOREST , bar_width, alpha = opacity, color = 'y'                   , label = 'R-Forest')


plt.xticks(x_axis + 2*bar_width, names)
#plt.grid( axis = 'both')

plt.tick_params(axis='y', labelsize=25)

plt.xlabel('Datasets', fontsize=35 )
plt.ylabel(' Validation Accuracy (%)', fontsize= 35)

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,  ncol=5, mode="expand", borderaxespad=0., prop={'size':30} )

plt.grid( axis = 'both')

plt.xlim( [-1,78] )
plt.show()
'''




Lenet_Val_acc  = [ 0.7428571582, 0.8142856956, 0.8000000119, 0.8142856956, 0.7857142687, 0.7714285851, 0.8000000119, 0.8142856956, 0.7857142687, 0.7857142687 ]
Lenet_Test_acc = [ 0.8225806355, 0.8064516187, 0.7741935253, 0.7096773982, 0.8064516187, 0.7903226018, 0.7903226018, 0.7419354916, 0.8064516187, 0.7580645084 ]


plt.show()

