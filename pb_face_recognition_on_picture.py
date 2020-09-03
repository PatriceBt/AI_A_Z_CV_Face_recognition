#Face Recognition

#Importing libraries
import cv2

#Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#ajout du smile
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') 

#Defining a function that will do the detections
def detect(gray, frame): #image 'frame' en noir et blanc (gray) et l'image en couleur 'frame'
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) #image en n&b, image reduced 1.3, 5 neighbor zones needs to be accepted - ok by experience
    #faces = (x,y) the position of the corner, (w,h) the weigh and high of rectangle

    #loop pour itérer chaque visage (il peut y en avoir plusieurs, oubien des false positives)
    for (x, y, w, h) in faces:
        #on va dessiner le rectangle pour le visage (en utilisant opencv qui a une fonction pour ça)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)  #3 et 4 = couleur et épaisseur
    
        #on peut maintenant détecter les yeux dans ce rectangle
     
        roi_gray = gray[y:y+h, x:x+w] #region of intérêt (intérieur du rectangle dans l'image n&b)
        roi_color= frame[y:y+h, x:x+w] #idem dans l'image en couleur
        #détection des yeux dans la roi gray
        #en utilisant la méthode multiscale
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        #on dessine le rectange des yeux (1 par oeil)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
        #et on détecte maintenant le ou les smiles (on ne sait jamais !)
        smile = smile_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (sx, sy, sw, sh) in smile:
            #et on dessine le rectange du smile
            cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    
    return frame #frame contient les rectangles dessinés
    #remarque : roi_frame est un sous-ensemble de frame donc ce qui est dessiné dedans l'est dans frame
    
#Doing some Face Recognition with the webcam

#on a besoin de la dernière image venant de la webcam 
#avec videocapture
#video_capture = cv2.VideoCapture(1) #0 si webcam internal, 1 sinon

#loop sur toute les images venant de la webcam
#arrêt avec la touche q qui lancera le break
while True:
#    _, frame = video_capture.read() #lecture de l'image (on ne prend pas le premier élément)
    frame =  cv2.imread(r'C:\Users\LENOVO\Pictures\Capture.PNG')
    print('image',frame)
    #on appel detect après avoir transformé en gray avec cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #average blue, red, greend pour avoir les bons contrasts
    canvas = detect(gray, frame) #canvas = image avec les rectangles !
    
    #affichage successifs des résultats avec imshow fonction
    #cv2.imshow('Video', canvas)
    cv2.imshow('image', canvas )
    cv2.waitKey(0)
    
    #stop si q est appuyé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("ok terminé")
        break
    
#turn off the webcam
video_capture.release()
#fermer la fenêtre d'affichage
cv2.destroyAllWindows()

    
