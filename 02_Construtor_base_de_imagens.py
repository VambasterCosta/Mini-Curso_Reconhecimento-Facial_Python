# codigo para construção das bases de imagens atraves de captura de video contendo o individuo para treinamento.

import numpy as np
import cv2
import os

#Gera uma lista de nomes dos individuos que serão utilizados para geração da base de imagens
def carregaNomes(local_arquivo):
    listaDeNomes = []
    pFile = open(local_arquivo, "r")
    for line in pFile:
        listaDeNomes.append(line.rstrip())
    return listaDeNomes


# cria diretorios para cada nome pertencente a lista de nomes, para guardas as imagens de cada individuo
def criaPastaComNomes(listaNomes):
    for nome in listaNomes:
        try:
            print("criou o diretório: " + nome)
            os.mkdir(nome)
        except OSError:
            print("Não foi possivel criar o diretório.")




def salvaFacesDetectadas(nome):
    face_cascade = cv2.CascadeClassifier('arquivos_essenciais/haarcascade_frontalface_default.xml')

   # cap = cv2.VideoCapture("videos_treinamento/"+nome + ".mp4") #inicia o video para a captura video de uma diretorio
    cap = cv2.VideoCapture(0) #inicia o video para a captura (webcam)

    cont = 0

    while(cont < 100): #captura até a quantidade de frames que queria utilizar para o trinamento da dos algoritmos
        print("foram encontradas "+cont+" faces")

        ret, img = cap.read()

        #se não possuir frame, exit
        if(ret == False):
            cap.release()
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# converte frame em escala de cinza
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) # define um limite de escala entre as features de haar(visinhos)

        #Se nenhuma face for encontrada continue
        if not np.any(faces):
            continue

        #se achar uma face no frame recorte ele
        for (x, y, w, h) in faces:
            rostoImg = img[y:y+h, x:x+w]

        #verifica o tamanho da face recortada, se for muito pequena desconsidera ela
        larg, alt, _ = rostoImg.shape
        if(larg * alt <= 20 * 20):
            continue

        #Se foi possivel passar por todos os processos, redimensiona a imagem e salva ela em seu diretorio
        rostoImg = cv2.resize(rostoImg, (255, 255))
        rostoImg = cv2.cvtColor(rostoImg, cv2.COLOR_BGR2GRAY)# converte frame em escala de cinza
        cv2.imwrite("base_treinamento/"+nome +"1."+ str(cont)+".jpg", rostoImg)
        cont += 1
            
    cap.release()

# função Principal Python
def main():
    
    #captura a quantidade de nomes dos individuos para treinamento da base. (treinamento supervisionado)
    listaNomes = carregaNomes("arquivos_essenciais/input.txt")
    
    #cria um diretório para cada nome contido na lista
    criaPastaComNomes(listaNomes)

    
    for nome in listaNomes:
        print("Analisando o video: " + nome)
        salvaFacesDetectadas(nome)


if __name__ == "__main__":
    main()
