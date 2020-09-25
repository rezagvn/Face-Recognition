import cv2
import os

#Memanggil library numpy untuk perhitungan
import numpy as np

#Memanggil PIL atau python image library untuk memasukan gambar
from PIL import Image

import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#Membuat penyimpanan lokal untuk binary histogram
recognizer = cv2.face.LBPHFaceRecognizer_create();

#Mengambil algoritma haarcascade yang untuk wajah
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

#Membuat method untuk mendapatkan gambar dan sample 
def getImagesAndLabels(path):

    #Mendapatkan alamat gambar dari library os atau oprating system
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    #Inisialisasi sample wajah
    faceSamples=[]
    
    #Inisialisasi ID dari dataset
    ids = []

    #Perulangan di dalam file 
    for imagePath in imagePaths:

        #Mendapatkan gambar lalu menkonversi menjadi abu abu
        PIL_img = Image.open(imagePath).convert('L')

        #Mengubah sample yang berupa gambar menjadi array
        img_numpy = np.array(PIL_img,'uint8')

        #Mendapatkan id sample dari folder dataset
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        #Mengambil gambar wajah dari direktori trainer
        faces = detector.detectMultiScale(img_numpy)

        #Perulangan di setiap sample, berdasarkan id mereka
        for (x,y,w,h) in faces:

            #Menambahkan gambar dari sample 
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            #Menambahkan id
            ids.append(id)

    #Melanjutkan ke antrian sample dan id selanjutnya
    return faceSamples,ids

#Mendapatkan id dan sample dari dataset
faces,ids = getImagesAndLabels('dataset')

#Mencoba model sistem dari sample wajah dan id
recognizer.train(faces, np.array(ids))

#Save model ke direktori trainer/trainer.yml
assure_path_exists('trainer/')
recognizer.save('trainer/trainer.yml')
