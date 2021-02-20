from Tkinter import *
import Tkinter
import tkMessageBox
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
root=Tk()
def real():
		#load model
		model = model_from_json(open("fer1.json", "r").read())
		#load weights
		model.load_weights('fer1.h5')


		face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


		cap=cv2.VideoCapture(0)

		while True:
		    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
		    if not ret:
			continue
		    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

		    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


		    for (x,y,w,h) in faces_detected:
			cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
			roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
			roi_gray=cv2.resize(roi_gray,(48,48))
			img_pixels = image.img_to_array(roi_gray)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 255

			predictions = model.predict(img_pixels)

			#find max indexed array
			max_index = np.argmax(predictions[0])

			emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
			predicted_emotion = emotions[max_index]

			cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

		    resized_img = cv2.resize(test_img, (1000, 700))
		    cv2.imshow('Facial emotion analysis ',resized_img)



		    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
			break

		cap.release()
		cv2.destroyAllWindows
root.geometry("500x420")
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.resizable(0,0)
root.title("Facial Expression Analysis Using AI")
frame.config(background="light blue")
label = Label(frame, text="Facial Expression Analysis Using AI",bg='light blue',font=('Times 18 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="brain.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)
but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,text="RealTime Analysis",command=real,font=('helvetica 15 bold'))
but1.place(x=5,y=150)



but4=Button(frame,padx=5,pady=5,width=5,bg='white',fg='black',relief=GROOVE,text='EXIT',command=root.quit,font=('helvetica 15 bold'))
but4.place(x=215,y=225)


root.mainloop()
