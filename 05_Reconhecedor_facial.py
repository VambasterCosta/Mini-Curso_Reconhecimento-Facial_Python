import cv2
import time

detectorFace = cv2.CascadeClassifier("arquivos_essenciais/haarcascade_frontalface_default.xml")

reconhecedorEigen = cv2.face.EigenFaceRecognizer_create()
reconhecedorEigen.read("arquivos_essenciais/classificadorEigen.yml")




reconhecedorLBPH  = cv2.face.LBPHFaceRecognizer_create()
reconhecedorLBPH.read("arquivos_essenciais/classificadorLBPH.yml")

largura, altura = 255, 255
font = cv2.FONT_HERSHEY_SIMPLEX
#camera = cv2.VideoCapture("video_teste/felipe.mp4")
camera = cv2.VideoCapture(0)


while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza,scaleFactor=1.5, minSize=(30,30))

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (255, 255))
        cv2.rectangle(imagem, (x+10, y+10), (x + l+10, y + a+10), (0,0,0), 1)
        cv2.putText(imagem, "Face Detect - viola e jones", (x, y + (a)), font, 0.5, (0,0,0))

        idLBPH, confiancaLBPH = reconhecedorLBPH.predict(imagemFace)

        idEigen, confiancaEigen = reconhecedorEigen.predict(imagemFace)

     
        nomeLBPH = ""
        if idLBPH == 1:
            nomeLBPH = 'LBPH - Felipe'
       
        if confiancaLBPH <= 27:
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,0,255), 1)
            cv2.putText(imagem, nomeLBPH, (x,y +(a+30)), font, 0.5, (0,0,255))
            cv2.putText(imagem, str(confiancaLBPH), (x,y + (a+45)), font, 0.4, (0,0,255))

        nomeEigen = ""
        if idEigen == 1:
            nomeEigen = 'Eigen - Felipe'
       
        if confiancaEigen <= 17000:
            cv2.rectangle(imagem, (x+5, y+5), (x + l+5, y + a+5), (0,255,0), 1)
            cv2.putText(imagem, nomeEigen, (x+5, y + (a+60)), font, 0.5, (0,255,0))
            cv2.putText(imagem, str(confiancaEigen), (x+5, y + (a+80)), font, 0.4, (0,255,0))
        #time.sleep(0.5)
        
        



    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == (27):
        break

camera.release()
cv2.destroyAllWindows()