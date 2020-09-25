import cv2

#mengambil algoritma haar cascade dari library open cv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#mendeklarasikan cap sebagai webcam yang jadi input
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    #membaca webcam
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #mendeteksi wajah menggunakan algoritma haar cascade
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #membuat bingkai kotak di luar wajah
    for (x, y, w, h) in faces:
        cv2.putText(img,"Reza",(x,y-10),font,0.75,(0,0,255),2,cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #memunculkan image yang didapat dari webcam
    cv2.imshow('img', img)
    #menstop pembacaan kamera jika huruf q pada keyboard dipencet
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
#menutup program
cap.release()
cv2.destroyAllWindows()  
