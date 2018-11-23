'''
Created on 14/06/2018

@author: robotica
'''

from cv2 import cvtColor
from scipy.io import savemat
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from skimage.morphology import label
from skimage.measure import perimeter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def extract_features( images ):

    feature=[]

    for img in images:
         
        Labimg=cvtColor(img,cv2.COLOR_RGB2Lab)
        ft=[]
        
        for canal in range(2):
            can=canal+1
            Labimg[:,:,can]=cv2.blur(Labimg[:,:,can],(5,5))
            umbrla,mascara = cv2.threshold(Labimg[:,:,can],0,255,cv2.THRESH_OTSU)
            if canal==1:
                mascara=mascara==0
            else:
                mascara=mascara==255
            Etiquetado=label(mascara,neighbors=4)
            Numero_de_Elementos=Etiquetado.max()
            pixeles=np.float64(mascara.sum())
            LabimgM=mascara*Labimg[:,:,can]
            #plt.imshow(LabimgM) 
            #plt.show() 
            CCM=greycomatrix(LabimgM,distances=[1,5],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],symmetric=True,normed =True)###symmetric 
            NdC=np.zeros((CCM.shape[2],CCM.shape[3]))
            for i in range(CCM.shape[3]):
                for j in range(CCM.shape[2]):
                    CCM[0,0,j,i]=0
                    NdC[j,i]=CCM[:,:,j,i].sum()            
            Compacidad=[]
            for i in range(Numero_de_Elementos):
                Elemento=Etiquetado==(i+1)
                Area=Elemento.sum()
                Perimetro=perimeter(Elemento)
                Compacidad.append(Perimetro**2/Area)
            CompacidadP=np.array(Compacidad).sum()/Numero_de_Elementos
            #contrast: 
            Contraste=greycoprops(CCM, prop='contrast') 
            Contraste=np.array(np.reshape(Contraste,8))/(pixeles*Numero_de_Elementos)  
            #dissimilarity: 
            dissimilarity=greycoprops(CCM, prop='dissimilarity')
            dissimilarity=np.array(np.reshape(dissimilarity,8))/(pixeles*Numero_de_Elementos) 
            #homogeneity: 
            homogeneity=greycoprops(CCM, prop='homogeneity')      
            homogeneity=np.array(np.reshape(homogeneity,8))/(pixeles*Numero_de_Elementos) 
            #energy: 
            energy=greycoprops(CCM, prop='energy')  
            energy=np.array(np.reshape(energy,8))/(pixeles*Numero_de_Elementos) 
            ft.extend(np.reshape(np.array([Contraste,dissimilarity,homogeneity,energy]).T,32).tolist())
            ft.append(Numero_de_Elementos)
            ft.append(CompacidadP)
            ft.append(np.array(Compacidad).sum())
        Gray=cvtColor(img,cv2.COLOR_RGB2GRAY )
        ###############################Gray#########################################
        ############################################################################
        Gray=cv2.blur(Gray,(5,5))
        umbrla,mascara = cv2.threshold(Gray,0,255,cv2.THRESH_OTSU)
        mascara=mascara==0
        Etiquetado=label(mascara,neighbors=4)
        Numero_de_Elementos=Etiquetado.max()
        pixeles=np.float64(mascara.sum())
        GrayM=mascara*Gray
        #plt.imshow(GrayM) 
        #plt.show() 
        CCM=greycomatrix(GrayM,distances=[1,5],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],symmetric=True,normed =True)###,symmetric=True,normed =True
        NdC=np.zeros((CCM.shape[2],CCM.shape[3]))
        
        for i in range(CCM.shape[3]):
            for j in range(CCM.shape[2]):
                CCM[0,0,j,i]=0
                NdC[j,i]=CCM[:,:,j,i].sum()    
        Compacidad=[]
        for i in range(Numero_de_Elementos):
            Elemento=Etiquetado==(i+1)
            Area=Elemento.sum()
            Perimetro=perimeter(Elemento)
            Compacidad.append(Perimetro**2/Area)
        CompacidadP=np.array(Compacidad).sum()/Numero_de_Elementos
    #contrast: 
        Contraste=greycoprops(CCM, prop='contrast') 
        Contraste=np.array(np.reshape(Contraste,8))/(pixeles*Numero_de_Elementos)  
    #dissimilarity: 
        dissimilarity=greycoprops(CCM, prop='dissimilarity')
        dissimilarity=np.array(np.reshape(dissimilarity,8))/(pixeles*Numero_de_Elementos) 
    #homogeneity: 
        homogeneity=greycoprops(CCM, prop='homogeneity')      
        homogeneity=np.array(np.reshape(homogeneity,8))/(pixeles*Numero_de_Elementos) 
    #energy: 
        energy=greycoprops(CCM, prop='energy')  
        energy=np.array(np.reshape(energy,8))/(pixeles*Numero_de_Elementos)
    
        ft.extend(np.reshape(np.array([Contraste,dissimilarity,homogeneity,energy]).T,32).tolist())
        ft.append(Numero_de_Elementos)
        ft.append(CompacidadP)
        ft.append(np.array(Compacidad).sum())    
       
        ###########################################################################
        ###########################################################################    
        feature.append(ft)
        
    
    feature=np.array(feature).T
    ###################################Normalizacion del conjunto#####################
    Promedio=np.mean(feature,1)
    Features=np.zeros(feature.shape)
    Maximos=np.zeros(feature.shape[0])

    for i in range(feature.shape[0]):
        Maximos[i]=feature[i,:].max()

    for i in range(feature.shape[1]):
        Features[:,i]=(feature[:,i]-Promedio)/Maximos 
    #################################################################################3

    return Features, Promedio, Maximos

#savemat('featureCR1_augmented',{'featureCR1':Feautures})
#savemat('Labels_augmented',{'Labels':labels})
#savemat('Normalizacion_augmented',{'Promedio':Promedio,'Maximos':Maximos})