import cv2
import os
import numpy as np
from PIL import Image
import time


def getImagemComId():
    caminhos = [os.path.join('Felipe', f) for f in os.listdir('Felipe')]
    cont = 0
    for caminhoImagem in caminhos:
       imagemFace = Image.open(caminhoImagem).convert('L')
       print(cont)
       imagemNP = np.array(imagemFace, 'uint8')
      
        
       cv2.imwrite("EscalaDeCinza/cris1." + str(cont)+".png", imagemNP)
       cont = cont + 1
       key = cv2.waitKey(1)
       if key == 27:
           breack

    #return np.array(ids), faces

getImagemComId()

print("Fim...")

