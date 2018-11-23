from skimage.io import imread
import matplotlib.pyplot as plt
import os
retval = os.getcwd()
print(retval)
DIRECTORIOS=['C1','C2']
labels=[]
feature=[]
for directorio in DIRECTORIOS:
    os.chdir(directorio)
    Imagenes=os.listdir()
    for imagen in Imagenes:
        print(imagen)
        img = imread(imagen)
        plt.imshow(img)
        plt.show()
        if directorio=='C1':
            if imagen[0]=='D':
                labels.append(1)
            else:
                
                labels.append(2)
        else:
            if imagen[0]=='E':
                labels.append(3)
            else:
                labels.append(4)
    os.chdir(retval)
"""
    D == Diestro    Etiqueta == 1
    P == Proestro   Etiqueta == 4
    M == Metaestro  Etiqueta == 2
    E == Estro      Etiqueta == 3
"""