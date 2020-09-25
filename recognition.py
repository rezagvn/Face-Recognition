#Memanggil library computer vision
import cv2

#Memanggil library numpy untuk perhitungan
import numpy as np

import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#Membuat penyimpanan lokal untuk binary histogram
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

#Memanggil program trainer.yml
recognizer.read('trainer/trainer.yml')

#Mengambil algoritma haarcascade yang untuk wajah
cascadePath = "haarcascade_frontalface_default.xml"

#Membuat klasifier yang digunakan dalam algoritma haar cascade 
faceCascade = cv2.CascadeClassifier(cascadePath);

#Set font yang akan digunakan
font = cv2.FONT_HERSHEY_SIMPLEX

#Inisialisasi webcam yang digunakan sebagai kamera
cam = cv2.VideoCapture(0)

#Perulangan yang akan dijalankan selama program berlangsung
while True:
    #Membaca frame dari webcam
    ret, im =cam.read()

    #Mengkonversi gambar yang didapat dari kamera menjadi abu abu
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    #Membaca semua wajah yang ada di video
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    #Perulangan yang digunakan untuk setiap wajah yang dibaca
    for(x,y,w,h) in faces:

        #Membuat kotak di sekitar wajah
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        #Untuk Mengenali wajah dari id
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        #Mengecek ID dari dataset
        if(Id == 1):
            Id = "Reza Geovani".format(round(100 - confidence, 2))
        elif(Id == 4):
            Id = "Unknown".format(round(100 - confidence, 2))
        else:
            print(Id)
            Id = "Unknown"

        #Membuat text nama yang ada di atas kotak
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

    #Menampilkan gambar yang didapat dari webcam
    cv2.imshow('im',im)

    # Jika 'q' dipencet, maka akan menutup program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop Kameranya
cam.release()

# Menutup Windows yang dibuka
cv2.destroyAllWindows()
