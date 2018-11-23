'''
Created on 17/04/2017

@author: robotica
'''

from matplotlib import style as styl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

    
import itertools

##=================================================================================================================
##=================================================================================================================
##=================================================================================================================


def plot_curves(x_l, x_c, font_size, scale = 'Normal'):
    plt.figure()
    
    if ( scale == 'semilog' ):
        plt.semilogy(n, x_l, 'r', label = 'Hyperplanes')
        plt.semilogy(n, x_c, 'b', label = 'Hyperboxes')
    if ( scale == 'normal'):
        plt.plot(n, x_l, 'r', label = 'Hyperplanes')
        plt.plot(n, x_c, 'b', label = 'Hyperboxes')
    
    if ( scale == 'loglog'):
        plt.plot(n, x_l, 'r', label = 'Hyperplanes')
        plt.plot(n, x_c, 'b', label = 'Hyperboxes')

    plt.grid( axis = 'both')

    plt.xlabel('Number of hyperplanes / hyperboxes',  fontsize= font_size )
    plt.ylabel('Max. response regions',  fontsize=font_size )
        
    
    
    plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., prop={'size':25} )
    
    
    plt.tick_params(axis='y', labelsize=20)
    plt.tick_params(axis='x', labelsize=20)

    plt.show()




################################################################################################
################################ PLOT   MLNN  VS  SUPERVISED CLUSTERING

names = ( 'XOR','Seeds', 'wine', 'Shuttle', 'OD', 'KM', 'ArChar', 'TicTacToe', 'LR', 'Vertebral', 'Whs', 'Statlog', 'MAGIC', 'Wilt', 'B', 'CCC', 'A', 'Arem', 'Tae', 'DR', 'MP-RSS', 'EEGE', 'WineQ'  ) 
 
HNN_FE = [ 1, 1, 1, 0.9986206897, 0.9906579689, 0.9862068892, 0.9850000143, 0.9765625,    0.9582499957, 0.9354838729, 0.9318181872, 0.8913043737, 0.8843322784, 0.8790000081, 0.8600000143, 0.822666663,  0.8125,       0.7800526818, 0.7741935253, 0.7575757504, 0.7344696916, 0.6602136194, 0.6112244725  ]

KNN = [ 1, 0.9761904762, 1, 0.993908046,  0.9777724088, 0.7793103448, 0.935,        0.7864583333, 0.7495,       0.8548387097, 0.8977272727, 0.8333333333, 0.8364879075, 0.656,        0.83,         0.807,        0.785,        0.7442528736, 0.5161290323,0.6796536797, 0.7261363636, 0.5644192256, 0.5663265306 ]

BAYES = [ 0.5, 0.9523809524, 1,            0.8743678161, 0.9832487718, 0.7931034483, 0.161,        0.703125,     0.64525,      0.7741935484, 0.8636363636, 0.8260869565, 0.7326498423, 0.634,        0.84,         0.6276666667, 0.78,         0.664032567, 0.5161290323, 0.5367965368, 0.6393939394, 0.5457276368, 0.4234693878 ]

D_TREE = [0.98, 0.9285714286, 0.83333, 0.9989655172, 0.9303374406, 0.8827586207, 0.975, 0.90625, 0.83175, 0.7903225806, 0.8863636364, 0.7463768116, 0.8322818086, 0.628, 0.75, 0.6936666667, 0.65, 0.7491618774, 0.6129032258, 0.683982684, 0.6916666667, 0.4953271028, 0.5408163265 ]

R_FOREST = [1, 0.9285714286, 1, 0.9991954023, 0.9661754047, 0.8275862069, 0.98, 0.828125, 0.9205, 0.8387096774, 0.8863636364, 0.847826087, 0.8703995794, 0.634, 0.79, 0.7181666667, 0.71, 0.7755028736, 0.7096774194, 0.7012987013, 0.7261363636, 0.4712950601, 0.6387755102  ]

style.use('ggplot')

x_axis = np.linspace(0 , 75, num = 23)

n_groups = 23
bar_width = 0.5
opacity = 0.8

plt.Figure

#plt.bar(x_axis               , HNN_FE  , bar_width, alpha = opacity, color = 'blue'            , label = 'HNN-FE')
#plt.bar(x_axis +   bar_width , HNN_C   , bar_width, alpha = opacity, color = 'royalblue'       , label = 'HNN-C')
#plt.bar(x_axis + 2*bar_width , DMN     , bar_width, alpha = opacity, color = 'mediumvioletred' , label = 'DMN')
#plt.bar(x_axis + 3*bar_width , DNN     , bar_width, alpha = opacity, color = 'mediumseagreen'  , label = 'DNN')
#plt.bar(x_axis + 4*bar_width , SVM_RBF , bar_width, alpha = opacity, color = 'peru'            , label = 'SVM-RBF')

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

################################################################################################
#################################  PLOT  NUMBER OR REGIONS

length = 2000
n = np.linspace(1, length, length, dtype=np.int)

x_l=[]
x_c = []
for i in range( len(n)):
    list_num = np.linspace( 1, n[i], n[i], dtype = np.int) 
    
    comb = itertools.combinations( list_num, 2)
    
    cont = 0
    
    for ii in list(comb):
        #print (ii)
        cont += 1
        
    x_l.append( cont + n[i]+1 )

    x_c.append( (2.0*n[i] * (int(n[i-1]))) +2 )

x_c[0] =2


font_size = 35
plot_curves(x_l, x_c, font_size, scale = 'normal' )
plot_curves(x_l, x_c, font_size, scale = 'semilog' )
plot_curves(x_l, x_c, font_size, scale = 'loglog' )


num_classes = 10

plt.figure()
x_classes = []
x_c = np.asanyarray(x_c)
for i in range (1,num_classes):
    x_classes.append( x_c * i )

    custom_label =  str(i) +' Classes'
    plt.plot(n, x_classes[i-1], 'r', label = custom_label )

plt.grid( axis = 'both')

plt.xlabel('Number of hyperboxes',  fontsize= font_size )
plt.ylabel('Max. response regions',  fontsize=font_size )
        
    
    
plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
ncol=5, mode="expand", borderaxespad=0., prop={'size':25} )
    
    
plt.tick_params(axis='y', labelsize=20)
plt.tick_params(axis='x', labelsize=20)

plt.show()
    #x.append( size(combnk(1:n(i),2),1) + n(i) + 1)


#y = n.*(n-1)+2;

#plt.plot(n,x,'b',n,y,'g')
##=================================================================================================================
##=================================================================================================================
##=================================================================================================================
##  3D PLOT NUMBER OF PARAMETERS    3D PLOT NUMBER OF PARAMETERS    3D PLOT NUMBER OF PARAMETERS

names = [ 'TicTacToe' , 'LR', 'WineQ', 'ArChar', 'MAGIC', 'Shuttle', 'EEGE', 'MP-RSS', 'Seeds', 'CCC', 'Arem', 'DR', 'OD', 'Whs', 'KM', 'Tae', 'Statlog', 'wine', 'Vertebral', 'Wilt' ]

x =  [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13, 14, 15, 16, 17, 18, 19, 20  ]
x = np.asanyarray(x)

HNN_FE_Params =  [ 22572, 18760, 15242, 8866, 8736, 5307, 4832, 4282, 3607, 3602, 3370, 1362, 746, 649, 620, 601, 422, 293, 142, 98 ] 
HNN_FE_Params = np.asanyarray( HNN_FE_Params )

HNN_C_Params =  [ 88565, 106973, 26208, 10488, 39754, 123841, 113829, 45628, 35084, 28642, 46386, 90239, 60172, 75068, 33246, 98073, 82366, 4760, 86330, 24619 ] 
HNN_C_Params = np.asanyarray( HNN_C_Params )

DNN_Params   =  [ 4616, 13775, 43430, 4005, 24831, 18245, 4835, 9051, 17119, 1198, 10083, 11179, 4100, 2099, 1184, 28210, 155, 5933, 37171, 4280 ] 
DNN_Params = np.asanyarray( DNN_Params )

MNN_Params   =  [ 1068, 29560, 9744, 6804, 18092, 976, 17500, 15140, 156, 27380, 47668, 2752, 780, 500, 500, 288, 840, 336, 284, 1084 ] 
MNN_Params = np.asanyarray( MNN_Params )

SVM_Params = [ 10404, 120256, 36564, 13068, 45080, 1197, 104874, 24648, 182, 223284, 117438, 15960, 1625, 511, 345, 475, 2394, 702, 654, 315     ]
SVM_Params = np.asanyarray ( SVM_Params  )

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


cs = ['blue'] * 20
ax.bar( x, HNN_FE_Params, zs=0, zdir='y', color=cs, alpha=0.8)

cs = ['g'] * 20
ax.bar( x, MNN_Params, zs= 2, zdir='y', color=cs, alpha=0.8)

cs = ['r'] * 20
ax.bar( x, DNN_Params, zs= 4, zdir='y', color=cs, alpha=0.8)

cs = ['y'] * 20
ax.bar(x, SVM_Params, zs = 6, zdir = 'y', color=cs, alpha=0.8)

cs = ['royalblue'] * 20
ax.bar( x, HNN_C_Params, zs= 8, zdir='y', color=cs, alpha=0.8)



x_axis  = np.linspace(0 , 20, num = len(MNN_Params) )
x_axis -= 0.5

plt.xticks(x_axis , names, rotation = 45)
plt.yticks([0, 2, 4, 6, 8] , ['MLNNs', 'DMNs', 'MLPs', 'SVM-RBF', 'LMNN'], rotation = -15, fontsize=15)

ax.set_zlabel('Number of parameters')
#ax.set_yscale('log')
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')

plt.show()


##=================================================================================================================
##=================================================================================================================
##=================================================================================================================
##  3D PLOT NUMBER OF PARAMETERS    3D PLOT NUMBER OF PARAMETERS    3D PLOT NUMBER OF PARAMETERS

x, y, z = axes3d.get_test_data()

x = np.full( (20,20),  [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13, 14, 15, 16, 17, 18, 19, 20  ])
y = np.full( (20,20),  [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13, 14, 15, 16, 17, 18, 19, 20  ])     

y = np.transpose( y)

HNN_FE_Params= np.full( (20,20), [ 22572, 18760, 15242, 8866, 8736, 5307, 4832, 4282, 3607, 3602, 3370, 1362, 746, 649, 620, 601, 422, 293, 142, 98 ] )

HNN_C_Params = np.full( (20,20), [ 88565, 106973, 26208, 10488, 39754, 123841, 113829, 45628, 35084, 28642, 46386, 90239, 60172, 75068, 33246, 98073, 82366, 4760, 86330, 24619 ] )

DNN_Params   = np.full( (20,20), [ 4616, 13775, 43430, 4005, 24831, 18245, 4835, 9051, 17119, 1198, 10083, 11179, 4100, 2099, 1184, 28210, 155, 5933, 37171, 4280 ] )

MNN_Params   = np.full( (20,20), [ 1068, 29560, 9744, 6804, 18092, 976, 17500, 15140, 156, 27380, 47668, 2752, 780, 500, 500, 288, 840, 336, 284, 1084 ]) 


style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')


ax1.plot_wireframe(x,y, HNN_FE_Params , rstride = 3, cstride = 3, color = 'blue' )
ax1.plot_wireframe(x,y, HNN_C_Params  , rstride = 3, cstride = 3, color = 'royalblue')
ax1.plot_wireframe(x,y, DNN_Params    , rstride = 3, cstride = 3, color = 'r')
ax1.plot_wireframe(x,y, MNN_Params    , rstride = 3, cstride = 3, color = 'g')
plt.show()

##=================================================================================================================
##=================================================================================================================
##=================================================================================================================
## NUMBER OF PARAMETERS          NUMBER OF PARAMETERS              NUMBER OF PARAMETERS 

names = ( 'TicTacToe' , 'LR', 'WineQ', 'ArChar', 'MAGIC', 'Shuttle', 'EEGE', 'MP-RSS', 'Seeds', 'CCC', 'Arem', 'DR', 'OD', 'Whs', 'KM', 'Tae', 'Statlog', 'wine', 'Vertebral', 'Wilt' )


HNN_FE_Params = [ 22572, 18760, 15242, 8866, 8736, 5307, 4832, 4282, 3607, 3602, 3370, 1362, 746, 649, 620, 601, 422, 293, 142, 98 ]

HNN_C_Params = [ 88565, 106973, 26208, 10488, 39754, 123841, 113829, 45628, 35084, 28642, 46386, 90239, 60172, 75068, 33246, 98073, 82366, 4760, 86330, 24619 ]

DNN_Params = [ 4616, 13775, 43430, 4005, 24831, 18245, 4835, 9051, 17119, 1198, 10083, 11179, 4100, 2099, 1184, 28210, 155, 5933, 37171, 4280 ]

MNN_Params = MNN_Params   =  [ 1068, 29560, 9744, 6804, 18092, 976, 17500, 15140, 156, 27380, 47668, 2752, 780, 500, 500, 288, 840, 336, 284, 1084 ] 


x_axis    = np.linspace(0 , 75, num = len(MNN_Params) )
n_groups  = len(MNN_Params)
bar_width = 0.5
opacity   = 0.8


plt.bar(x_axis               , HNN_FE_Params  , bar_width, alpha = opacity, color = 'blue'  , label = 'HNN-FE')
plt.bar(x_axis +   bar_width , HNN_C_Params   , bar_width, alpha = opacity, color = 'royalblue'    , label = 'HNN-C')
plt.bar(x_axis + 2*bar_width , MNN_Params     , bar_width, alpha = opacity, color = 'r'            , label = 'DMN')
plt.bar(x_axis + 3*bar_width , DNN_Params     , bar_width, alpha = opacity, color = 'g'            , label = 'DNN')
plt.yscale('log')
plt.grid( axis = 'both')

plt.xticks(x_axis + 2*bar_width, names)

plt.tick_params(axis='y', labelsize=25)

plt.xlabel('Datasets', fontsize=35 )
plt.ylabel(' Number of Parameters', fontsize= 35)

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0., prop={'size':30} )

plt.show()
#plt.bar(x_axis + 4*bar_width , SVM_RBF , bar_width, alpha = opacity, color = 'y'            , label = 'SVM-RBF')


##=================================================================================================================
##=================================================================================================================
##=================================================================================================================


names = ( 'XOR', 'Seeds','wine','Shuttle', 'SP_5L','SP_3D','OD','KM','ArChar','TicTacToe','LR','Vertebral','Whs','Statlog','MAGIC','Wilt','B','CCC','A','Arem','Tae','DR','MP-RSS','EEGE','WineQ' ) 

HNN_FE = [ 1, 1, 1, 0.9986206897, 0.9982000051,0.9950000048,0.9906579689,0.9862068892, 0.9850000143,0.9765625,0.9582499957,0.9354838729,0.9318181872,0.8913043737,0.8843322784,0.8790000081, 0.8600000143,0.822666663,0.8125,0.7800526818, 0.7741935253,0.7575757504,0.7344696916,0.6602136194, 0.6112244725]

HNN_C = [ 1, 1, 1, 0.9964367816, 0.9964000022, 0.9950000048,0.9907385037, 0.9931034446, 0.986, 0.9427083135, 0.9637500067, 0.9516128898, 0.9545454383, 0.913043499,0.8845951612, 0.7059999704, 0.8566, 0.822, 0.81, 0.7804118772, 0.8064516187, 0.8, 0.7409090909, 0.6311748922, 0.6397958994 ]

DNN = [ 1, 0.9761904989, 0.3611111244, 0.7813793088, 0.995, 0.992, 0.7740194897, 0.9931034565, 0.9, 0.6614583582, 0.9504999986, 0.8709677573, 0.7613636181, 0.7753623249, 0.9499999967, 0.6260000223, 0.846, 0.7731666667, 0.805, 0.7397030651, 0.6774193741, 0.7792207792, 0.7431818182, 0.5507343124, 0.521428586 ]

DMN = [ 1, 1, 0.916667, 0.99954, 0.997, 0.987, 0.9691, 0.8, 0.931, 0.958333, 0.751537, 0.887097, 0.909091, 0.811594, 0.814542, 0.78629, 0.8233, 0.778259, 0.795, 0.725744, 0.75, 0.662338, 0.706626, 0.790721, 0.631974 ]

SVM_RBF = [ 1, 1, 0.9722222222, 0.9991954023, 0.9988, 0.9925, 0.9892083434, 0.9448275862, 0.978, 0.9791666667, 0.97325, 0.8225806452, 0.9204545455, 0.8188405797, 0.8688222923, 0.654, 0.8366666667, 0.773, 0.795, 0.7643678161, 0.4838709677, 0.7705627706, 0.7276515152, 0.5574098798, 0.6653061224 ]


x_axis = np.linspace(0 , 75, num = 25)

n_groups = 25
bar_width = 0.5
opacity = 0.8

plt.Figure

#plt.bar(x_axis               , HNN_FE  , bar_width, alpha = opacity, color = 'blue'            , label = 'HNN-FE')
#plt.bar(x_axis +   bar_width , HNN_C   , bar_width, alpha = opacity, color = 'royalblue'       , label = 'HNN-C')
#plt.bar(x_axis + 2*bar_width , DMN     , bar_width, alpha = opacity, color = 'mediumvioletred' , label = 'DMN')
#plt.bar(x_axis + 3*bar_width , DNN     , bar_width, alpha = opacity, color = 'mediumseagreen'  , label = 'DNN')
#plt.bar(x_axis + 4*bar_width , SVM_RBF , bar_width, alpha = opacity, color = 'peru'            , label = 'SVM-RBF')

plt.bar(x_axis               , HNN_FE  , bar_width, alpha = opacity, color = 'blue'             , label = 'HNN-FE')
plt.bar(x_axis +   bar_width , HNN_C   , bar_width, alpha = opacity, color = 'royalblue'        , label = 'HNN-C')
plt.bar(x_axis + 2*bar_width , DMN     , bar_width, alpha = opacity, color = 'r'                , label = 'DMN')
plt.bar(x_axis + 3*bar_width , DNN     , bar_width, alpha = opacity, color = 'g'                , label = 'DNN')
plt.bar(x_axis + 4*bar_width , SVM_RBF , bar_width, alpha = opacity, color = 'y'                   , label = 'SVM-RBF')


plt.xticks(x_axis + 2*bar_width, names)
plt.grid( axis = 'both')

plt.tick_params(axis='y', labelsize=25)

plt.xlabel('Datasets', fontsize=35 )
plt.ylabel(' Validation Accuracy (%)', fontsize= 35)


plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0., prop={'size':30} )
           
#plt.ylim( [0,1.05] )
#plt.xlim( [-1,25] )

plt.show()


##=================================================================================================================
##=================================================================================================================
##=================================================================================================================

# DMN vs DNN   Model Params   20 real life datasets
#names = ( 'WineQ','Vertebral','Tae','MAGIC','Shuttle','Seeds','LR','DR','Arem','MP-RSS', 'Wine','EEGE','TicTacToe','Wilt', 'OD', 'ArChar','Whs','CCC','KM','Statlog' )
#DNN_params_20_datasets = [ 43430,37171,28210,24831,18245,17119,13775,11179,10083,9051,5933,4835,4616,4280,4100,4005,2099,1198,1184,155   ]
#DMN_params_20_datasets = [ 9744, 284,288,18092,976,156,29560,2752,47668,15140,336,17500,1068,1084,780,6804,500,27380,500,840   ]

#names = ('Arem','LR','CCC','MAGIC','EEGE','MP-RSS','WineQ', 'ArChar','DR','Wilt','TicTacToe','Shuttle','Statlog', 'OD','Whs','KM', 'Wine','Tae','Vertebral','Seeds' )
names = ('Seeds','Vertebral','Tae', 'Wine', 'KM','Whs', 'OD','Statlog','Shuttle','TicTacToe','Wilt','DR', 'ArChar','WineQ','MP-RSS','EEGE','MAGIC','CCC','LR','Arem')

#DNN_params_20_datasets = [ 10083,13775,1198,24831,4835,9051,43430,4005,11179,4280,4616,18245,155,4100,2099,1184,5933,28210,37171,17119    ]
DNN_params_20_datasets = [ 17119,37171,28210,5933,1184,2099,4100,155,18245,4616,4280,11179,4005,43430,9051,4835,24831,1198,13775,10083 ]


#DMN_params_20_datasets = [47668,29560,27380,18092,17500,15140,9744,6804,2752,1084,1068,976,840,780,500,500,336,288,284,156   ]

DMN_params_20_datasets = [ 156, 284,288,336,500,500,780,840,976,1068,1084,2752,6804,9744,15140,17500,18092,27380,29560,47668      ]

x_axis = np.linspace(0 , 40, num = 20)

n_groups = 20
bar_width = 0.5
opacity = 0.8

plt.Figure

plt.bar(x_axis               , DNN_params_20_datasets      , bar_width, alpha = opacity, log=True,  color = 'r'     , label = 'DNN')
plt.bar(x_axis  +   bar_width, DMN_params_20_datasets     , bar_width, alpha = opacity,  log=True,  color = 'b'     , label = 'DMN')


plt.grid( axis = 'both')
plt.xticks(x_axis + bar_width, names)
plt.tick_params(axis='y', labelsize=38)
plt.tick_params(axis='x', labelsize=13)

plt.xlabel('Datasets', fontsize=55 )
plt.ylabel(' Number of Params ', fontsize= 55)
#plt.legend( prop={'size':18})


plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0., prop={'size':30} )
           
#plt.ylim( [0,48000] )
#plt.ticklabel_format(style='sci', axis='y')
plt.xlim( [-1,42] )

plt.show()


###====================================================================
## Accuracy Validation 
#names = ( 'KM', 'Seeds', 'LR','MAGIC', 'ArChar', 'Vertebral', 'Shuttle', 'DR', 'Statlog', 'OD', 'CCC', 'Whs', 'MP-RSS', 'Arem', 'Tae', 'TicTacToe', 'Wilt', 'EEGE', 'WineQ',  'Wine' )
names = ( 'Seeds','Shuttle','OD','TicTacToe','ArChar','wine','Whs','Vertebral','MAGIC','Statlog','KM','EEGE','Wilt','CCC','LR','Tae','Arem','MP-RSS','DR','WineQ' )

#DNN_val_acc = [0.9931034565, 0.9761904989,0.9504999986,0.9499999967,0.9,0.8709677573,0.7813793088,0.7792207792,0.7753623249,0.7740194897,0.7731666667,0.7613636181,0.7431818182,0.7397030651,0.6774193741,0.6614583582,0.6260000223,0.5507343124,0.521428586,0.3611111244 ]

DNN_val_acc = [0.976190499, 0.781379309,0.77401949,0.661458358,0.9,0.361111124,0.761363618,0.870967757,0.949999997,0.775362325,0.993103457,0.550734312,0.626000022,0.773166667,0.950499999,0.677419374,0.739703065,0.743181818,0.779220779,0.521428586 ]

#DMNN_val_acc = [0.8,1,0.751537,0.814542,0.931,0.887097,0.99954,0.662338,0.811594,0.9691,0.778259,0.909091,0.706626,0.725744,0.75,0.958333,0.78629,0.790721,0.631974,0.916667]            
DMNN_val_acc = [ 1,0.99954,0.9691,0.958333,0.931,0.916667,0.909091,0.887097,0.814542,0.811594,0.8,0.790721,0.78629,0.778259,0.751537,0.75, 0.725744,0.706626,0.662338,0.631974 ]


x_axis = np.linspace(0 , 40, num = 20)

#DNN_trn_acc = np.asarray( DNN_trn_acc )
#DMNN_trn_acc = np.asarray( DMNN_trn_acc )

n_groups = 20
bar_width = 0.5
opacity = 0.8

plt.Figure
plt.bar(x_axis              , DNN_val_acc   , bar_width, alpha = opacity, color = 'r', label = 'DNN')
plt.bar(x_axis + bar_width  , DMNN_val_acc  , bar_width, alpha = opacity, color = 'b', label = 'DMN')

plt.grid( axis = 'both')
plt.xticks(x_axis + bar_width, names)
plt.tick_params(axis='y', labelsize=38)
plt.tick_params(axis='x', labelsize=14)

plt.xlabel('Datasets', fontsize=57 )
plt.ylabel('Validation Accuracy (%)', fontsize=57 )
plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., prop={'size':30} )

plt.xlim( [-1,42] )
plt.ylim( [0,1.05] )

plt.show()

###===================================================================================================
###===================================================================================================
###===================================================================================================
# Dataset  Dimentionality vs time 

dimentionality = [ 4, 5,5,6,6,6,7,7,7,8,9, 9,11,12,13,14,15,16,20,24 ]
                   

DNN_time = [23.37343407,1.22711587,2.829027891,70.93055105,9.905077934,2.565799952,10.20815802,30.53052092,3.705388069,3.18801713,80.59380507,1.550594091,36.25815892,8.955138922,2.766165018,4.433340073,31.24886107,55.18968892,6.340955973,43.81688809 ]

MNN_time = [179.674544,0.162497,0.118234,920.197676,0.108297,0.25243,10.763055,0.137698,0.094806,0.118752,0.140803,0.361743,38.47805,41.637967,0.108631,0.449635,290.543476,87.876776,4.597605,1345.920196]

max_DNN = np.max( DNN_time )
max_MNN = np.max( MNN_time )

max_y = np.max(  [max_DNN , max_MNN ])

x = list(range(21))
x = x[1 :]

poly_indx_DNN_1 = np.poly1d(np.polyfit(x, DNN_time, 1))

poly_indx_MNN_1 = np.poly1d(np.polyfit(x, MNN_time, 1))


xp = np.linspace(0, 20, 20)
  

plt.Figure

plt.scatter(x, DNN_time, marker='*', color = 'r', s= 50 )
plt.plot(xp, poly_indx_DNN_1(xp), linestyle='--', color = 'r',  label = 'DNN', linewidth=6.0 )


plt.scatter(x, MNN_time, marker='o', color = 'b', s= 50)
plt.plot(xp, poly_indx_MNN_1(xp), linestyle='-', color = 'b', label = 'DMN', linewidth=6.0 )

plt.grid( axis = 'both')
plt.tick_params(axis='both', labelsize=38)

plt.xlabel('Number of features',  fontsize=57 )
plt.ylabel('Training time (Sec.)',  fontsize=57 )

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., prop={'size':30} )

plt.yscale('log')
plt.ylim( [-0.1,max_y] )
plt.xlim( [0,20] )

plt.show()



MNN_time = np.asanyarray( MNN_time )
DNN_time = np.asanyarray( DNN_time )

#Repito: haciendo que los tiempos sean medidos en escala logaritmica (sacando el log10 de los tiempos), entonces
#se aplica la regresion. Ese problema se eliminaria.

log_MNN_time = np.log10( MNN_time )
log_DNN_time = np.log10( DNN_time )


poly_indx_DNN_2 = np.poly1d(np.polyfit(x, log_DNN_time, 2))

poly_indx_MNN_2 = np.poly1d(np.polyfit(x, log_MNN_time, 2))


plt.Figure

plt.scatter(x, log_DNN_time, marker='*', color = 'r' , s= 50 )
plt.plot(xp, poly_indx_DNN_2(xp), linestyle='--', color = 'r',  label = 'DNN' , linewidth=6.0  )


plt.scatter(x, log_MNN_time, marker='o', color = 'b', s= 50 )
plt.plot(xp, poly_indx_MNN_2(xp), linestyle='-', color = 'b', label = 'DMN', linewidth=6.0 )

plt.grid( axis = 'both')
plt.tick_params(axis='both', labelsize=38)

plt.xlabel('Number of features',  fontsize=57 )
plt.ylabel('Training time (Sec.)',  fontsize=57 )

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., prop={'size':30} )

plt.yscale('log')
plt.ylim( [-0.01,max_y] )
plt.xlim( [0,20] )

plt.show()



###===================================================================================================
###===================================================================================================
###===================================================================================================
# Dataset  Dimentionality vs time 

dimentionality = [ 4, 5,5,6,6,6,7,7,7,8,9, 9,11,12,13,14,15,16,20,24 ]
                   

DNN_params = [9051, 1184,28210,10083,4280,37171,4005,4100,17119,2099,18245,4616,24831,43430,5933,155,4835,13775,11179,1198  ]
MNN_params = [15140, 500,288,47668,1084,284,6804,780,156,500,976,1068,18092,9744,336,840,17500,29560,2752,27380]

max_DNN = np.max( DNN_params )
max_MNN = np.max( MNN_params )

max_y = np.max(  [max_DNN , max_MNN ])

x = list(range(21))
x = x[1 :]

poly_indx_DNN_1 = np.poly1d(np.polyfit(x, DNN_params, 1))

poly_indx_MNN_1 = np.poly1d(np.polyfit(x, MNN_params, 1))


xp = np.linspace(0, 20, 20)
  
plt.Figure


plt.scatter(x, DNN_params, marker='*', color = 'r', s= 50 )
plt.plot(xp, poly_indx_DNN_1(xp), linestyle='--', color = 'r',  label = 'DNN', linewidth=6.0 )


plt.scatter(x, MNN_params, marker='o', color = 'b', s= 50)
plt.plot(xp, poly_indx_MNN_1(xp), linestyle='-', color = 'b', label = 'DMN', linewidth=6.0 )


plt.grid( axis = 'both')
plt.tick_params(axis='both', labelsize=38 )

plt.xlabel('Number of features',  fontsize=57 )
plt.ylabel('Num. of params ',  fontsize=57 )

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., prop={'size':30} )

plt.yscale('log')
plt.xlim( [0,20] )

plt.show()



poly_indx_DNN_2 = np.poly1d(np.polyfit(x, DNN_params, 2))

poly_indx_MNN_2 = np.poly1d(np.polyfit(x, MNN_params, 2))


plt.Figure


plt.scatter(x, DNN_params, marker='*', color = 'r' , s= 50 )
plt.plot(xp, poly_indx_DNN_2(xp), linestyle='--', color = 'r',  label = 'DNN' , linewidth=6.0  )


plt.scatter(x, MNN_params, marker='o', color = 'b', s= 50 )
plt.plot(xp, poly_indx_MNN_2(xp), linestyle='-', color = 'b', label = 'DMN', linewidth=6.0 )

plt.grid( axis = 'both')
plt.tick_params(axis='both', labelsize=38)

plt.xlabel('Number of features',  fontsize=57 )
plt.ylabel('Num. of params',  fontsize=57 )

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., prop={'size':30} )

plt.yscale('log')
plt.ylim( [-50,max_y] )
plt.xlim( [0,20] )
plt.show()


##=================================================================================================================
##=================================================================================================================
##=================================================================================================================
#DNN Validation single layer

names = ('1', '2', '3', '4', '5', '6', '7', '8',  '9', '10')


DNN_acc_val_2_10_spiral =  [ 0.9945000016,0.9964000002,0.7676,0.6151999995,0.5931999998,0.5840999999,0.5764,0.4860000223,0.4600002229,0.4522292 ]
DMNN_acc_val_2_10_spiral = [0.9924,0.9936,0.994,0.9972,0.9948,0.9965,0.9973,0.9955,0.9967,0.9975 ]

DNN_TanH_acc_val_2_10_spiral = [0.994, 0.9960000001,0.7855999986,0.6700000024,0.6139999983,0.5910000002,0.5700800027,0.568793557,0.5561333341,0.5473333338 ]
DNN_UT_acc_val_2_10_spiral    = [0.9920000003,0.8003999996,0.6596000027,0.6183999986,0.5956000008,0.5736999994,0.5532000003,0.5504000032,0.5420666672,0.5351333333]

#x_axis = range( len(DNN_trn_acc_2_10_spiral) )

x_axis = np.linspace(0 , 22, num = 10)

n_groups = 20
bar_width = 0.5
opacity = 0.8

plt.Figure
plt.bar(x_axis               , DMNN_acc_val_2_10_spiral     , bar_width, alpha = opacity, color = 'b'     , label = 'DMN')
plt.bar(x_axis +   bar_width , DNN_acc_val_2_10_spiral      , bar_width, alpha = opacity, color = 'r'     , label = 'TLPReLU')
plt.bar(x_axis + 2*bar_width , DNN_TanH_acc_val_2_10_spiral , bar_width, alpha = opacity, color = 'g'     , label = 'TLPtanh')
plt.bar(x_axis + 3*bar_width , DNN_UT_acc_val_2_10_spiral    , bar_width, alpha = opacity, color = 'yellow', label = 'TLPsiglin')


plt.grid( axis = 'both')
plt.xticks(x_axis + 2*bar_width , names)

plt.tick_params(axis='both', labelsize=38)

plt.xlabel('No. Loops', fontsize=55 )
plt.ylabel(' Validation Accuracy (%)', fontsize= 55)
#plt.legend( prop={'size':18})

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0., prop={'size':30} )
           
plt.ylim( [0,1.05] )
plt.xlim( [-1,25] )

plt.show()



##=================================================================================================================
##=================================================================================================================
##=================================================================================================================
#DNN train single layer

names = ('1', '2', '3', '4', '5', '6', '7', '8',  '9', '10')


DNN_trn_acc_2_10_spiral =  [ 0.9945000016,0.9971000002, 0.7179000017,0.6129000028, 0.5903999967,0.5843499934,0.5682400026,0.453803427, 0.538034264,0.423803849 ]

DMNN_trn_acc_2_10_spiral = [0.9968, 0.9972, 0.9964, 0.9977, 0.999, 0.9974, 0.9976, 0.9978,0.9976, 0.9987 ]

DNN_TanH_trn_acc_2_10_spiral = [0.994, 0.9960000001,0.7855999986,0.6700000024,0.6139999983,0.5910000002,0.5700800027,0.568793557,0.5561333341,0.5473333338 ]
DNN_UT_tr_acc_2_10_spiral    = [0.9920000003,0.8003999996,0.6596000027,0.6183999986,0.5956000008,0.5736999994,0.5532000003,0.5504000032,0.5420666672,0.5351333333]

#x_axis = range( len(DNN_trn_acc_2_10_spiral) )

x_axis = np.linspace(0 , 22, num = 10)

n_groups = 20
bar_width = 0.5
opacity = 0.8

plt.Figure
plt.bar(x_axis               , DMNN_trn_acc_2_10_spiral     , bar_width, alpha = opacity, color = 'b'     , label = 'DMN')
plt.bar(x_axis +   bar_width , DNN_trn_acc_2_10_spiral      , bar_width, alpha = opacity, color = 'r'     , label = 'DNNReLU')
plt.bar(x_axis + 2*bar_width , DNN_TanH_trn_acc_2_10_spiral , bar_width, alpha = opacity, color = 'g'     , label = 'DNNtanh')
plt.bar(x_axis + 3*bar_width , DNN_UT_tr_acc_2_10_spiral    , bar_width, alpha = opacity, color = 'yellow', label = 'DNNsiglin')


plt.grid( axis = 'both')
plt.xticks(x_axis + 2*bar_width , names)

plt.tick_params(axis='both', labelsize=38)

plt.xlabel('No. Loops', fontsize=55 )
plt.ylabel(' Training Accuracy (%)', fontsize= 55)
#plt.legend( prop={'size':18})

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0., prop={'size':30} )
           
plt.ylim( [0,1.05] )
plt.xlim( [-1,25] )

plt.show()



###===================================================================================================
###===================================================================================================
###===================================================================================================
# Dataset  Dimentionality


#dimentionality = [ 4.00, 5,5,6,6,7,7,7,8,9,9,10,11,12,13,14,15,16,20,24 ]
dimentionality = [ 4, 5,5,6,6,6,7,7,7,8,9, 9,11,12,13,14,15,16,20,24 ]
DNN_Acc_Val = [0.743181818, 0.993103456,0.677419374,0.739703065,0.870967757,0.9,0.77401949,0.976190499,0.761363618,0.626000022,0.661458358,0.781379309,0.949999997,0.521428586,0.361111124,0.775362325,0.550734312,0.950499999,0.779220779,0.773166667  ]
MNN_Acc_Val = [0.706626,0.8,0.75,0.725744,0.887097,0.931,0.9691,1,0.909091,0.78629,0.958333,0.99954,0.814542,0.631974,0.916667,0.811594,0.790721,0.751537,0.662338,0.778259 ]



x = list(range(21))
x = x[1 :]

poly_indx_DNN_1 = np.poly1d(np.polyfit(x, DNN_Acc_Val, 1))

poly_indx_MNN_1 = np.poly1d(np.polyfit(x, MNN_Acc_Val, 1))


xp = np.linspace(0, 20, 20)
  
plt.Figure


plt.scatter(x, DNN_Acc_Val, marker='*', color = 'r', s= 50 )
plt.plot(xp, poly_indx_DNN_1(xp), linestyle='--', color = 'r',  label = 'DNN', linewidth=6.0 )


plt.scatter(x, MNN_Acc_Val, marker='o', color = 'b', s= 50)
plt.plot(xp, poly_indx_MNN_1(xp), linestyle='-', color = 'b', label = 'DMN', linewidth=6.0 )

plt.grid( axis = 'both')
plt.tick_params(axis='both', labelsize=38)

plt.xlabel('Number of features',  fontsize=57 )
plt.ylabel('Validation Accuracy (%)',  fontsize=57 )

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., prop={'size':30} )

plt.ylim( [0,1.05] )
plt.xlim( [0,20] )

plt.show()



poly_indx_DNN_2 = np.poly1d(np.polyfit(x, DNN_Acc_Val, 2))

poly_indx_MNN_2 = np.poly1d(np.polyfit(x, MNN_Acc_Val, 2))


plt.Figure


plt.scatter(x, DNN_Acc_Val, marker='*', color = 'r' , s= 50 )
plt.plot(xp, poly_indx_DNN_2(xp), linestyle='--', color = 'r',  label = 'DNN' , linewidth=6.0  )


plt.scatter(x, MNN_Acc_Val, marker='o', color = 'b', s= 50 )
plt.plot(xp, poly_indx_MNN_2(xp), linestyle='-', color = 'b', label = 'DMN', linewidth=6.0 )

plt.grid( axis = 'both')
plt.tick_params(axis='both', labelsize=38)

plt.xlabel('Number of features',  fontsize=57 )
plt.ylabel('Validation Accuracy (%)',  fontsize=57 )

plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., prop={'size':30} )

plt.ylim( [0,1.05] )
plt.xlim( [0,20] )

plt.show()





#plt.plot(xp, poly_indx_DNN_2(xp), linestyle=':',  color = 'r' )

#plt.plot(xp, poly_indx_MNN_1(xp), linestyle='--', marker='o', color = 'b' )
#plt.plot(xp, poly_indx_MNN_2(xp), linestyle=':', marker='o', color = 'b' )

#plt.plot(xp, poly_indx_MNN_2(xp), linestyle=':' ,  color = 'b' )

#plt.scatter( dimentionality, DNN_Acc_Val,  marker='*', color='r')
#plt.scatter( dimentionality, MNN_Acc_Val,  marker='*', color='b')











##=========================================================================================
#DNN vall single layer
names = ('1', '2', '3', '4', '5', '6', '7', '8',  '9', '10')
DNN_val_acc_2_10_spiral =  [0.9945000016, 0.9964000002,0.7676,0.6151999995,0.5931999998,0.5840999999,0.5764,0.6260000223,0.6260000223,0.6260000223 ]
DMNN_val_acc_2_10_spiral = [0.9924, 0.9936, 0.994, 0.9972, 0.9948, 0.9965, 0.9973, 0.9955, 0.9967, 0.9975  ]


x_axis = np.linspace(0 , 20, num = 10)

n_groups = 20
bar_width = 0.5
opacity = 0.8

plt.Figure
plt.bar(x_axis              , DNN_val_acc_2_10_spiral   , bar_width, alpha = opacity, color = 'r', label = 'DNN')
plt.bar(x_axis + bar_width  , DMNN_val_acc_2_10_spiral  , bar_width, alpha = opacity, color = 'b', label = 'DMN')

plt.grid( axis = 'both')
plt.xticks(x_axis + bar_width, names )
plt.tick_params(axis='both', labelsize=38)

plt.xlabel('No. Loops',  fontsize=57 )
plt.ylabel(' Validation Accuracy (%)',  fontsize= 57)
plt.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., prop={'size':30} )

plt.ylim( [0,1.05] )
plt.xlim( [-1,22] )


plt.show()



###===================================================================================================
###===================================================================================================
names = ( 'LR', 'Wilt','KM','MAGIC','Vertebral','ArChar','Seeds','Tae','Whs','OD','Shuttle','DR','CCC','MP-RSS','Arem','Statlog','TicTacToe','EEGE','WineQ','wine')

#DNN train Real Datasets
DNN_trn_acc = [0.9839999926,0.9829453804,0.9767441819,0.9736250087,0.9395161146,0.9000000057,0.8452381087,0.8333333284,0.7926136565,0.7871791733,0.7847701165,0.7826086957,0.78025,0.7711471062,0.7395000748,0.7010869697,0.6514360402,0.5513184246,0.5079122078,0.4084507181 ]
DMNN_trn_acc =[ 0.993041,1,1,0.959638,0.919355,0.9968,0.982143,1,0.997159,0.9997,1,0.992366,0.935242,0.935055,0.954144,0.952899,1,0.960364,0.984835,1 ]

#DNN_trn_acc = DNN_trn_acc * 100 
#DMNN_trn_acc = DMNN_trn_acc* 100
 
x_axis = np.linspace(0 , 40, num = 20)

DNN_trn_acc = np.asarray( DNN_trn_acc )
DMNN_trn_acc = np.asarray( DMNN_trn_acc )

n_groups = 20
bar_width = 0.5
opacity = 0.8

plt.Figure
plt.bar(x_axis              , DNN_trn_acc   , bar_width, alpha = opacity, color = 'r', label = 'DNN')
plt.bar(x_axis + bar_width  , DMNN_trn_acc  , bar_width, alpha = opacity, color = 'b', label = 'DMN')

plt.grid( axis = 'both')
plt.xticks(x_axis + bar_width / 2, names)

plt.xlabel('Datasets')
plt.ylabel('Training Accuracy (%)')
plt.legend()

plt.show()





print('Done')



print ( styl.available )


