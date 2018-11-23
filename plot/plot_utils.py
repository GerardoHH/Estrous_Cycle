'''
Created on 24/08/2017

@author: robotica
'''
from plot.laplotter import LossAccPlotter 

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.io import loadmat
from scipy.constants.constants import alpha

from mpl_toolkits.mplot3d import  Axes3D

import itertools

import imageio



def my_plot_train_loss( history):

    plt.figure(1)
    plt.subplot(211)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')

    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    

    plt.subplot(212)
    
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.show()

def plot_confusion_matrix(cm, num_classes, title = 'Confusion Matrix', cmap = plt.cm.Blues, normalize = False , save = False, path_save = None):

    plt.Figure
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange( len(num_classes))
    plt.xticks(tick_marks, num_classes, rotation=45)
    plt.yticks(tick_marks, num_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if (save):
        plt.savefig(path_save)

def plot_dataset_and_boxes( P, W, path_save = None):
    W = np.asanyarray( W )
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for idx in range(0, W.shape[1]):
        d_x_W_min = W[0][idx]
        d_x_W_max = W[1][idx]
            
        #                     X    Y    width  height        
        p = patches.Rectangle(  (d_x_W_min[0], d_x_W_min[1]),   d_x_W_max[0],   d_x_W_max[1],   hatch='/', fill=False )
        ax.add_patch(p)
        
    
    half = int(P.shape[0]/2)
    ax.scatter(P[0: half, 0], P[0: half, 1]  , cmap=plt.cm.Spectral, s=1)
    ax.scatter(P[half:half*2, 0], P[half:half*2, 1]  , cmap=plt.cm.Spectral, s=1)
        
    plt.grid(axis= 'both')
    
    if (path_save == None):
        plt.show()
    else:
        #plt.savefig(path_save, format='pdf')
        plt.savefig(path_save, format='jpg')
        plt.close()
    

def load_evaluated_mesh( mesh_path ):
    global Y, XX, YY, error
    
    dict = loadmat( mesh_path )
    
    Y  = np.asanyarray(dict['Y'])
    XX = np.asanyarray(dict['X1'])
    YY = np.asanyarray(dict['X2'])
    
    XX = np.asarray(XX, dtype = np.float64)
    YY = np.asarray(YY, dtype = np.float64)
    
    error = dict['Etest']
    
    P = dict['P']
    
    Ptest = dict['Ptest']
    
    P = P.transpose()
    
    Ptest = P.transpose()
    # Y[np.where(Y == 2)] = 0
    
    return Y, XX,YY, error, P, Ptest
    print("Mesh Loaded ....")

def plot_from_matlab( P_ori, mesh_path , path_save , class_div):
    Y, XX,YY, _,P, Ptest =  load_evaluated_mesh( mesh_path )
    
    plt.contour(YY, XX, Y, colors='k')
    
    if ( class_div == 2 ):
        half = int(P.shape[0]/2)
        plt.scatter(P[0: half, 0], P[0: half, 1]  , cmap=plt.cm.Spectral, s=1)
        plt.scatter(P[half:half*2, 0], P[half:half*2, 1]  , cmap=plt.cm.Spectral, s=1)
    
    if (class_div == 3 ):
        half = int(P.shape[0]/3)
        half_2 = int( 2*half )
        half_3 = int( 3*half )
        plt.scatter(P[0: half, 0], P[0: half, 1]  , cmap=plt.cm.Spectral, s=1)
        plt.scatter(P[half:half_2, 0], P[half:half_2, 1]  , cmap=plt.cm.Spectral, s=1)
        plt.scatter(P[half_2:half_3, 0], P[half_2:half_3, 1]  , cmap=plt.cm.Spectral, s=1)
    
    plt.grid( axis='both' )
    plt.xlim(-5,6)
    plt.ylim(-5,6)
    
    if (path_save == None):
        plt.show()
    else:
        plt.savefig(path_save + 'desc_boundary_DC.pdf', format='pdf')
        plt.close()
    

def plot_train_loss( hist, path_save = None ):
    
    plotter = LossAccPlotter( show_regressions=False, show_loss_plot= False)
        
    for i in range( len(hist.history['loss']) ):
                    plotter.add_values( i ,
                    loss_train= hist.history['loss'][i],
                    loss_val  = hist.history['val_loss'][i],
                    acc_train = hist.history['acc'][i],
                    acc_val  =  hist.history['val_acc'][i],
                    redraw=False)
         
    plotter.redraw()

    if ( path_save == None ):
        plotter.block()
    else:
        plotter.save_plot( path_save )
        plotter.close()


def plot_train_No_loss( hist, path_save = None ):
    
    plotter = LossAccPlotter( show_regressions=False, show_loss_plot= False)
        
    for i in range( len(hist.history['loss']) ):
                    plotter.add_values( i ,
                    loss_train=  0,
                    loss_val  = 0,
                    acc_train = hist.history['acc'][i],
                    acc_val  =  hist.history['val_acc'][i],
                    redraw=False)
         
    plotter.redraw()

    if ( path_save == None ):
        plotter.block()
    else:
        plotter.save_plot( path_save )
        plotter.close()


def plot_decision_boundary_2_class_3D(P, model, batch_size, h = 0.05, half_dataset = False, path_save = None, expand = 0.0):
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    zmin, zmax = P[:, 2].min(), P[:, 2].max()
    
    xmin = xmin + (xmin - 5)*expand
    xmax = xmax + (xmax + 10)*expand
    
    ymin = ymin + (ymin - 10)*expand
    ymax = ymax + (ymax + 5)*expand
    
    #zmin = zmin + (zmin -2)*expand
    #zmax = zmax + (zmax +2)*expand
    zmin = zmin + (zmin )*expand
    zmax = zmax + (zmax )*expand
    
    #dx, dy, dz = (xmax - xmin)*0.1, (ymax - ymin)*0.1 , (zmax - zmin)*0.1
    
     #create mesh 
    xx, yy, zz = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h),
                         np.arange(zmin, zmax, h))
    
    print(" Predicting .... ")
    
    xx = xx.ravel()
    yy = yy.ravel()
    zz =  zz.ravel()
    
    
    ## procesar por capas 
    '''
    zz_uniq_val = np.unique( zz )

    for zz_val in zz_uniq_val:
        ## indices  de zz_val
        idx_of_val = np.where( zz == zz_val)
        
        xx_tmp = xx[ idx_of_val ]
        yy_tmp = yy[ idx_of_val ]
        zz_tmp = zz[ idx_of_val ]
         
        z = model.predict(np.c_[xx_tmp , yy_tmp, zz_tmp] , batch_size ) # default batch size 32
        
        z_class = []
        z_c = []
        z_idx = []
    
        ran = 0.15 #  percentage
        sup = 0.5 + 0.5*ran
        inf = 0.5 - 0.5*ran
    
        for idx in range(len( z ) ):
            if ( z[idx, 0] >= z[idx, 1] ):
                z_class.append( 0 )
             
            if ( z[idx, 1] > z[idx, 0] ):
                z_class.append( 1 )
        
            if ( (inf <= z[idx][0] and z[idx][0] <= sup) or
                  (inf <= z[idx][1] and z[idx][1] <= sup) ):
                 z_c.append( [xx_tmp[idx] , yy_tmp[idx], zz_tmp[idx] ])
                 z_idx.append( idx )
    
        print(" Done predictiing ... ")
        
        
        ax = Axes3D( plt.figure() )
        ax.scatter( xx_tmp, yy_tmp, zz_tmp, color = 'red', alpha=0.2 )
        ax.scatter( z_c[0], z_c[1], z_c[2], color = 'black', alpha=0.2 )
        plt.draw()
        
        #ax.scatter(P[0:mid, 0], P[0:mid, 1], P[0:mid, 2] , color = 'red')
        #ax.scatter(P[mid+1:2*mid, 0], P[mid+1:2*mid, 1], P[mid+1:2*mid, 2], color = 'blue'  )
    
    
        
        #z_c   = np.asanyarray(z_c  , dtype= np.float32)
        #z_idx = np.asanyarray(z_idx, dtype= np.int)
    
        #z_t = np.array( z_class, dtype = np.float32 )
        #z_t = z_t.reshape(xx.shape)
        
    
    '''
    
    
    z = model.predict(np.c_[xx , yy, zz] , batch_size ) # default batch size 32
    
    z_class = []
    z_c = []
    z_idx = []
    
    ran = 0.15 #  percentage
    sup = 0.5 + 0.5*ran
    inf = 0.5 - 0.5*ran
    
    for idx in range(len(z) ):
        if ( z[idx, 0] >= z[idx, 1] ):
            z_class.append( 0 )
             
        if ( z[idx, 1] > z[idx, 0] ):
            z_class.append( 1 )
        
        if ( (inf <= z[idx][0] and z[idx][0] <= sup) or
             (inf <= z[idx][1] and z[idx][1] <= sup) ):
            z_c.append( [xx[idx] , yy[idx], zz[idx] ])
            z_idx.append( idx)
    
    print(" Done predictiing ... ")
    
    z_c   = np.asanyarray(z_c  , dtype= np.float32)
    z_idx = np.asanyarray(z_idx, dtype= np.int)
    
    z_t = np.array( z_class, dtype = np.float32 )
    #Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z_t = z_t.reshape(xx.shape)
    

    
    
    z_class = np.asanyarray(z_class, dtype=np.float32)
    
    c_1_idx = np.where( z_class == 0 )
    c_2_idx = np.where( z_class == 1 )
    
    mid = int(len(P)/2)
    
    ##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK
    ##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK
    
    #ax = Axes3D( plt.figure() )
    #ax.scatter( xx[c_1_idx], yy[c_1_idx], zz[c_1_idx], color = 'red', alpha=0.1 )
    #ax.scatter(P[0:mid, 0], P[0:mid, 1], P[0:mid, 2] , color = 'red')
    #ax.scatter( xx[c_2_idx], yy[c_2_idx], zz[c_2_idx], color = 'blue', alpha=0.1 )
    #ax.scatter(P[mid+1:2*mid, 0], P[mid+1:2*mid, 1], P[mid+1:2*mid, 2], color = 'blue'  )
    
    #ax.set_xlabel('X ')
    #ax.set_ylabel('Y ')
    #ax.set_zlabel('Z ')

    #ax.view_init(90, 45)
    #plt.draw()
    #plt.savefig(path_save + '10C_desc_boundary_'+ str(90) +'.pdf', format='pdf')
    
    
    #filenames = []
    #for angle in range(0, 180, 2):
    #    ax.view_init(angle, 45)
    #    plt.draw()
    #    plt.pause(.001)
    #    plt.savefig(path_save + '10C_desc_boundary_'+ str(angle) +'.jpg', format='jpg')
    #    filenames.append( path_save + '10C_desc_boundary_'+ str(angle) +'.jpg' )
    
    #images = []    
    #for filename in filenames:
    #    images.append(imageio.imread(filename))
    
    #imageio.mimsave(path_save+ '10C_animation.gif', images)
    
    
    ##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK
    ##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK##### OKOKOKOK
        
        

    
    
   
    
    #########################################################################################
    #########################################################################################
    ########################   ANIMACION  SI SIRVE      #####################################
    
    #ax.view_init(26, 45)
    #plt.draw()
    #plt.savefig(path_save + 'desc_boundary_'+ str(90) +'.pdf', format='pdf')
    
    #ax.view_init(78, 45)
    #plt.draw()
    #plt.savefig(path_save + 'desc_boundary_'+ str(78) +'.pdf', format='pdf')
    
    #filenames = []
      

    #########################################################################################
    #########################################################################################
    #########################################################################################
    
    #ax.scatter(P[0:mid, 0], P[0:mid, 1], P[0:mid, 2] , color = 'red')
    #ax.scatter(P[mid+1:2*mid, 0], P[mid+1:2*mid, 1], P[mid+1:2*mid, 2], color = 'blue'  )
    
    #ax2 = Axes3D( plt.figure() )
    #ax2.scatter(z_c[:,0], z_c[:,1], z_c[:,2], color='black', s=1)
    
    
    plot_des_boundary_spiral3D( z, z_class, mid, xx, yy, zz, P, z_c, 0.30, 0.70, path_save, save = True )
    
    
    plt.grid(axis= 'both')
    
    if (path_save == None):
        plt.show()
    else:
        plt.savefig(path_save + 'desc_boundary.pdf', format='pdf')
        plt.close()

def plot_des_boundary_spiral3D( z, z_class, mid, xx, yy, zz, P, z_c, inf, sup, path_save, save = False ):
        c_1_idx = np.where( z_class == 0 )
        c_2_idx = np.where( z_class == 1 )
    
        db_c_1 = np.where(  np.logical_and( z[c_1_idx][0] > inf, z[c_1_idx][0] < sup ))
        db_c_2 = np.where(  np.logical_and( z[c_2_idx][1] > inf, z[c_2_idx][1] < sup ))
    
        ax3 = Axes3D( plt.figure() )
        ax3.scatter(P[0:mid, 0], P[0:mid, 1], P[0:mid, 2] , color = 'red')
        ax3.scatter(P[mid+1:2*mid, 0], P[mid+1:2*mid, 1], P[mid+1:2*mid, 2], color = 'blue'  )
    
    
        ax3.scatter(xx[db_c_1], yy[db_c_1], zz[db_c_1], color='black', s=1, alpha =0.9 )
        ax3.scatter(xx[db_c_2], yy[db_c_2], zz[db_c_2], color='black', s=1, alpha =0.9)
        ax3.scatter(z_c[:,0], z_c[:,1], z_c[:,2], color='black', s=1, alpha =0.9)
    
        ax3.set_xlabel('X ')
        ax3.set_ylabel('Y ')
        ax3.set_zlabel('Z ')
        
        ax3.view_init(16, 45)
        plt.draw()
        plt.savefig(path_save + '11C_desc_boundary_'+ str(16) +'.pdf', format='pdf')
    
            
        filenames = []
        for angle in range(0, 180, 2):
            ax3.view_init(angle, 45)
            plt.draw()
            plt.pause(.001)
            plt.savefig(path_save + '11C_desc_boundary_'+ str(angle) +'.jpg', format='jpg')
            filenames.append( path_save + '11C_desc_boundary_'+ str(angle) +'.jpg' )
    
        images = []    
        for filename in filenames:
            images.append(imageio.imread(filename))
        
        imageio.mimsave(path_save+ '11C_animation.gif', images)
        
def plot_decision_boundary_2_class_sklearn(P, model, h = 0.05, half_dataset = False, path_save = None, expand = 0.0):
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    
    xmin = xmin + xmin*expand
    xmax = xmax + xmax*expand
    
    ymin = ymin + ymin*expand
    ymax = ymax + ymax*expand
    
    #dx, dy = (xmax - xmin)*0.1, (ymax - ymin)*0.1
    
     #create mesh 
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()]  ) # default batch size 32
    

    z_class = []
    
    for idx in range(len(z) ):
        if ( z[idx] == 1 ):
            z_class.append( 0 )
             
        if ( z[idx] == 2 ):
            z_class.append( 1 )
    
    z_t = np.array( z_class, dtype = np.float32 )
    #Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z_t = z_t.reshape(xx.shape)
    
    
    plt.contour(xx, yy, z_t, colors='k')#,levels=[0.48, 0.52])  #, 
    #plt.contourf(xx, yy, z_t, cmap=plt.cm.Paired, alpha=0.8)  #,
    #Original
    
    if ( half_dataset ):
        half = int(P.shape[0]/2)
        plt.scatter(P[0: half, 0], P[0: half, 1]  , cmap=plt.cm.Spectral, s=1)
        plt.scatter(P[half:half*2, 0], P[half:half*2, 1]  , cmap=plt.cm.Spectral, s=1)
    else:
        plt.scatter(P[:, 0], P[:, 1], cmap=plt.cm.Spectral, s=1)
    
    plt.grid(axis= 'both')
    plt.xlim(-18,50)
    plt.ylim(-5,80)
        
    if (path_save == None):
        plt.show()
    else:
        plt.savefig(path_save + 'desc_boundary.pdf', format='pdf')
        plt.close()
    
    
def plot_decision_boundary_3_class_sklearn(P, model, h = 0.05, half_dataset = False, path_save = None, expand = 0.0):
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    
    xmin = xmin + xmin*expand
    xmax = xmax + xmax*expand
    
    ymin = ymin + ymin*expand
    ymax = ymax + ymax*expand
    
    #dx, dy = (xmax - xmin)*0.1, (ymax - ymin)*0.1
    
     #create mesh 
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()]  ) # default batch size 32
    

    z_class = []
    
    for idx in range(len(z) ):
        if ( z[idx] == 1 ):
            z_class.append( 0 )
             
        if ( z[idx] == 2 ):
            z_class.append( 1 )
            
        if ( z[idx] == 3 ):
            z_class.append( 2 )
    
    z_t = np.array( z_class, dtype = np.float32 )
    #Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z_t = z_t.reshape(xx.shape)
    
    
    plt.contour(xx, yy, z_t, colors='k')#,levels=[0.48, 0.52])  #, 
    #plt.contourf(xx, yy, z_t, cmap=plt.cm.Paired, alpha=0.8)  #,
    #Original
    
    if ( half_dataset ):
        half = int(P.shape[0]/3)
        plt.scatter(P[0: half, 0], P[0: half, 1]  , cmap=plt.cm.Spectral, s=1)
        plt.scatter(P[half:half*2, 0], P[half:half*2, 1]  , cmap=plt.cm.Spectral, s=1)
        plt.scatter(P[half*2:half*3, 0], P[half*2:half*3, 1]  , cmap=plt.cm.Spectral, s=1)
    else:
        plt.scatter(P[:, 0], P[:, 1], cmap=plt.cm.Spectral, s=1)
    
    plt.grid(axis= 'both')
    
    
    if (path_save == None):
        plt.show()
    else:
        plt.savefig(path_save + 'desc_boundary.pdf', format='pdf')
        plt.close()

def plot_decision_boundary_2_class(P, model, batch_size, h = 0.05, half_dataset = False, path_save = None, expand = 0.0):
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    
    xmin = xmin + xmin*expand
    xmax = xmax + xmax*expand
    
    ymin = ymin + ymin*expand
    ymax = ymax + ymax*expand
    
    #dx, dy = (xmax - xmin)*0.1, (ymax - ymin)*0.1
    
     #create mesh 
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))
    
    z = model.predict(np.c_[xx.ravel(), yy.ravel()] , batch_size ) # default batch size 32

    z_class = []
    
    for idx in range(len(z) ):
        if ( z[idx, 0] >= z[idx, 1] ):
            z_class.append( 0 )
             
        if ( z[idx, 1] > z[idx, 0] ):
            z_class.append( 1 )
    
    z_t = np.array( z_class, dtype = np.float32 )
    #Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z_t = z_t.reshape(xx.shape)
    
    
    plt.contour(xx, yy, z_t, colors='k')#,levels=[0.48, 0.52])  #, 
    #plt.contourf(xx, yy, z_t, cmap=plt.cm.Paired, alpha=0.8)  #,
    #Original
    
    if ( half_dataset ):
        half = int(P.shape[0]/2)
        plt.scatter(P[0: half, 0], P[0: half, 1]  , cmap=plt.cm.Spectral, s=1)
        plt.scatter(P[half:half*2, 0], P[half:half*2, 1]  , cmap=plt.cm.Spectral, s=1)
    else:
        plt.scatter(P[:, 0], P[:, 1], cmap=plt.cm.Spectral, s=1)
    
    plt.grid(axis= 'both')
    plt.xlim( -6, 6)
    plt.ylim( -6, 6)
    
    if (path_save == None):
        plt.show()
    else:
        #plt.savefig(path_save , format='pdf')
        plt.savefig(path_save + 'desc_boundary_model.pdf', format='pdf')
        plt.close()
    
    
def plot_decision_boundary_3_class(P, model, batch_size, h = 0.05, half_dataset = False, path_save = None, expand = 0.0, dashed = False):
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    
    xmin = xmin + xmin*expand
    xmax = xmax + xmax*expand
    
    ymin = ymin + ymin*expand
    ymax = ymax + ymax*expand
    
    dx, dy = (xmax - xmin)*0.1, (ymax - ymin)*0.1
    
     #create mesh 
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))
    
    z = model.predict(np.c_[xx.ravel(), yy.ravel()] , batch_size ) # default batch size 32
    
    z_t_0 = np.array( z[:,0] , dtype = np.float32 )
    z_t_1 = np.array( z[:,1] , dtype = np.float32 )
    z_t_2 = np.array( z[:,2] , dtype = np.float32 )
    
    z_t_0 = z_t_0.reshape(xx.shape)
    z_t_1 = z_t_1.reshape(xx.shape)
    z_t_2 = z_t_2.reshape(xx.shape)
    
    
    if ( dashed ):
        c_0 = plt.contour(xx, yy, z_t_0, colors='r', levels=[0.49, 0.51], alpha = 0.25 )
        #for c in c_0.collections:
        #    c.set_dashes([(0, (1.0, 12.0))])
    
        c_1 = plt.contour(xx, yy, z_t_1, colors='g', levels=[0.49, 0.51], alpha = 0.8  )
        for c in c_1.collections:
            c.set_dashes([(2, (3.0, 5))])
            
        c_2 = plt.contour(xx, yy, z_t_2, colors='b', levels=[0.49, 0.51], alpha = 0.8 ,  )
        for c in c_2.collections:
            c.set_dashes([(4, (3.0, 5))]) 
    else:
        c_0 = plt.contour(xx, yy, z_t_0, colors='r', levels=[0.49, 0.51])
            
        c_1 = plt.contour(xx, yy, z_t_1, colors='g', levels=[0.49, 0.51])
            
        c_2 = plt.contour(xx, yy, z_t_2, colors='b', levels=[0.49, 0.51])
     
    
    
    if ( half_dataset ):
        half = int(P.shape[0]/3)
        plt.scatter(P[0: half, 0], P[0: half, 1]  , cmap=plt.cm.Spectral, s=1, color = 'r', alpha = 0.2)
        plt.scatter(P[half:half*2, 0], P[half:half*2, 1]  , cmap=plt.cm.Spectral, s=1, color = 'g', alpha = 0.2)
        plt.scatter(P[half*2:half*3, 0], P[half*2:half*3, 1]  , cmap=plt.cm.Spectral, s=1, color = 'b', alpha = 0.2)
    else:
        plt.scatter(P[:, 0], P[:, 1], cmap=plt.cm.Spectral, s=1)
    
    plt.grid(axis= 'both')
    plt.xlim( -7.5, 10.0)
    plt.ylim( -7.5, 10.0)
    
    if (path_save == None):
        plt.show()
    else:
        plt.savefig(path_save + 'desc_boundary.pdf', format='pdf')
        plt.close()