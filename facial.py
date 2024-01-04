import cv2

# Chargement du modèle de reconnaissance faciale
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Démarrage de la capture vidéo à partir de la caméra du smartphone
cap = cv2.VideoCapture(0)

while True:
    # Lecture de la vidéo image par image
    ret, frame = cap.read()

    # Conversion de l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages dans l'image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Dessin d'un rectangle autour de chaque visage détecté
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Affichage de l'image avec les rectangles autour des visages
    cv2.imshow('frame',frame)

    # Arrêt du programme si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()
