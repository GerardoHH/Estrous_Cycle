'''
Created on Oct 12, 2018

@author: gerardo
'''

import  numpy as np

import  matplotlib.pyplot as plt

import math


def ViewActivations ( ):
    
    
    files = [
             '/home/gerardo/activaciones/LetterRecognition.npy',
             '/home/gerardo/activaciones/XOR.npy',
             '/home/gerardo/activaciones/A.npy',
             '/home/gerardo/activaciones/B.npy',
             '/home/gerardo/activaciones/CreditCardClients.npy',
             '/home/gerardo/activaciones/ARem.npy',
             '/home/gerardo/activaciones/ArtifitialChar.npy',
             '/home/gerardo/activaciones/DiabeticRetinopathy.npy',
             '/home/gerardo/activaciones/EEGEyeState.npy',
             '/home/gerardo/activaciones/MovementPredictionRSS.npy',
             '/home/gerardo/activaciones/MAGICGammaTelescope.npy',
             '/home/gerardo/activaciones/OccupancyDetection.npy',
             '/home/gerardo/activaciones/Shuttle.npy',
             '/home/gerardo/activaciones/Wilt.npy',
             '/home/gerardo/activaciones/Vertebral.npy',
             '/home/gerardo/activaciones/KnoledgeModeling.npy',
             '/home/gerardo/activaciones/Seeds.npy',
             '/home/gerardo/activaciones/Wholesale.npy',
             '/home/gerardo/activaciones/Wine.npy',
             '/home/gerardo/activaciones/winequality.npy',
             '/home/gerardo/activaciones/tictaetoe.npy',
             '/home/gerardo/activaciones/TeachingAssitant.npy',
             '/home/gerardo/activaciones/statlog.npy',
             '/home/gerardo/activaciones/espiral_2C_5L.npy',
             '/home/gerardo/activaciones/3D_spiral.npy',
             '/home/gerardo/activaciones/MNIST.npy' ]
    
    
    #files = [
    #         '/home/gerardo/activaciones/LetterRecognition.npy' ]
             
    title_subplot = ['LR', 'XOR', 'A', 'B', 'CCC', 'Arem', 'ArChar', 'DR', 'EEGE', 'MP-RSS', 'MAGIC', 'OD', 'Shuttle', 'Wilt',
                     'Vertebral', 'KM', 'Seeds', 'Whs', 'Wine', 'WineQ', 'TicTacToe', 'Tae', 'Statlog', 'SP 5L', 'SP 3D', 'MNIST' ]
    x_subplot     = []
    y_sub_plot    = []
    
    idx_title = 0
    for file_name in files:
        
        activations = np.load( file_name)
        
        for idx in range(0, activations.shape[0] ):
            one_indeces = np.where( activations[idx] >= 0 )
            cero_indeces = np.where( activations[idx] < 0 )

            activations[idx][one_indeces]  = 1
            activations[idx][cero_indeces] = 0

        total_neurons = activations.shape[1]
        
        acum_activations =  np.sum( activations, axis = 1)
        
        
        zero_proba = len( np.where (  acum_activations == 0.0 )[0] )
        one_proba  = len( np.where (  acum_activations == 1.0 )[0] )
        
        activations_proba = len( np.where ( acum_activations >= 2 )[0] )
        
        max_proba  = len( np.where (  acum_activations == float(total_neurons) )[0] )
        
        avrg_neuron_activation = len(  np.where( acum_activations > 0 )[0])
        
        
        x   = []
        val = np.zeros_like( np.arange(total_neurons) )
        
        for idx in range ( 0 , total_neurons ):
            x.append( idx )
            
            a = np.where ( acum_activations == idx )
            #val.append(  a[0])
            val[idx] = len(a[0])
            
        
        val = np.asanyarray(val )
 
        

        print("\n" + str(file_name))
        print("\n\t\t Total Neurons " + str( activations.shape[1]))
        print("\n\t\t Total Active " + str(  avrg_neuron_activation ))
        
        print("\n\t\t\t  Zero Activations proba : " + str( zero_proba        / activations.shape[0] ))
        print("\n\t\t\t  One Activations proba  : " + str( one_proba         / activations.shape[0] ))
        print("\n\t\t\t  Activation proba       : " + str( activations_proba / activations.shape[0] ))
    
        #print("\n\t\t\t  Maximum Activation proba : " + str( max_proba  / activations.shape[0]))
        
        #print("\n\t\t\t Total Activation proba: " + str(avrg_neuron_activation / activations.shape[0]) ) 
        
        print(" Total activations: " +str( activations.shape[0] ))
        
        
        
                    
        '''
        for idx in range( int(np.min(acum_activations)), int(np.max(acum_activations)) ):
            a = np.where ( acum_activations == idx )
            x.append(   idx )
            val.append( len(a[0]) )
        '''
        #x = np.linspace(1,150, 150, dtype = int)

        f = plt.figure()
                
        plt.bar(x, val, align = 'center')
        plt.grid(True)
        plt.title( title_subplot[idx_title] )
        plt.xlim( left = -0.5, right= total_neurons )
        idx_title = idx_title +1
        
        #xposition = [rango, 2*rango]
        #for xc in xposition:
        #    plt.axvline(x=xc, color='red', linestyle='--')
    
        #x = np.linspace(0,322, 323, dtype = np.int)
        #plt.plot(x, val, 'r--', linewidth=1)
        
        #plt.show()
        
        f.savefig("/home/gerardo/activaciones/figures_0_idex/" + str(file_name[ file_name.rfind('/')+1 : file_name.rfind('.')]) +".pdf")
        
        plt.close()

        x_subplot.append( x )
        y_sub_plot.append( val)
        print("\n ####################################################")
            
    
    rows = 9
    cols =  3
    f, axarr = plt.subplots(rows, cols)
    
    l_row = 0
    for idx in range(  0 , len(title_subplot)):
        print(" coordinates [" + str(l_row)+ "," + str( idx%cols ) +" ]   idx " + str(idx))
        axarr[l_row, idx%cols ].bar(x_subplot[idx], y_sub_plot[idx], align = 'center')
        
        axarr[l_row, idx%cols ].set_title( title_subplot[idx])
        axarr[l_row, idx%cols ].grid(True)
        
        if ( (idx%cols) == (cols -1) ):
            l_row = l_row + 1
        
        #axarr[0, 1].scatter(x, y)
        #axarr[0, 1].set_title('Axis [0,1]')
        #axarr[1, 0].plot(x, y ** 2)
        #axarr[1, 0].set_title('Axis [1,0]')
        #axarr[1, 1].scatter(x, y ** 2)
        #axarr[1, 1].set_title('Axis [1,1]')        
    
    f.savefig("/home/gerardo/activaciones/figures_0_idex/All_in_one.pdf")
    
    plt.close()
    
    
def main():
    
    ViewActivations()
    
if __name__ == "__main__":
    main()