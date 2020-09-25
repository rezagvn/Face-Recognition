import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#Inisialisasi vid cam yang diambil dari webcam
vid_cam = cv2.VideoCapture(0)

#Menangkap object muka dalam video menggunakan algoritma haarcascade 
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Memberikan id 1 pada muka yang ditangkap
face_id = 4

#Memberi input hitungan awal = 0
count = 0

assure_path_exists("dataset/")

#Looping yang akan berjalan sampai program berakhir
while(True):

    #Mengaktifkan webcam 
    _, image_frame = vid_cam.read()

    #Mengkonversi frame yang didapat webcam menjadi abu abu
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    #Mendeteksi beberapa wajah sekaligus dalam frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    #Perulangan di setiap wajah
    for (x,y,w,h) in faces:

        #Membuat kotak di wahag
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)

        #Menambahkan hitungan + 1
        count += 1

        #Mensave gambar yang didapat ke folder dataset
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        #Menampilkan kotak yang sudah dibuat di sekitar wajah
        cv2.imshow('frame', image_frame)

    #Jika key 'q' dipencet maka program akan berakhir
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    #Jika gambar yang didapat sudah mencapai 101, atau hitungan awal sudah mencapai 101 maka program akan berhenti
    elif count>100:
        break

#Stop video
vid_cam.release()

#Menutup semua windows
cv2.destroyAllWindows()
