# import sys
# import logging as log
# import datetime as dt
# from time import sleep
# import cv2
# import os
# import numpy as np
# import face_recognition
# from datetime import datetime 
# cascPath = "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
# log.basicConfig(filename='webcam.log',level=log.INFO)
# url="http://192.168.1.234:8080/shot.jpg"






# path='imgs'
# images=[]
# className=[]
# myList=os.listdir(path)
# for c in myList:
#     newImg=cv2.imread(f'{path}/{c}')
#     images.append(newImg)
#     className.append(os.path.splitext(c)[0])
# print(className)
# def finEncode(images):
#     encodeList=[]
#     for img in images:
#         img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         encode=face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# def addtofile(name):
#     with open("attendaneName.csv","r+")as f:
#         datalist=f.readlines()
#         namelist=[]
#         for l in datalist:
#             exsistname=l.split(',')
#             namelist.append(exsistname[0])
#         if name not in namelist:
#             now =datetime.now()
#             dateformat=now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dateformat}')

# encodeImages=finEncode(images)
# print("encodeing Done")


# cam=cv2.VideoCapture(url)
# while True:
#     success,img=cam.read()
#     imgSmall=cv2.resize(img,(0, 0),None,0.25,0.25)
#     imgSmall=cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

#     faceLoc=face_recognition.face_locations(imgSmall)
#     faceEncode=face_recognition.face_encodings(imgSmall,faceLoc)
#     for loc,encode in zip(faceLoc,faceEncode):
#         match=face_recognition.compare_faces(encodeImages,encode)
#         dist=face_recognition.face_distance(encodeImages,encode)
#         print(dist)
#         indx = np.argmin(dist)
#         if match[indx]:
#             name=className[indx].upper()
#         else:
#             name="unknown"
#         y1 , x2, y2 , x1=  loc
#         y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
#         cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#         cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#         cv2.putText(img ,name ,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
#         addtofile(name)
#     while True:
#         img_url=np.array(bytearray(urllib.request.urlopen(url).read()),dtype=np.uint8)
#         img=cv2.imdecode(img_url,-1)
#         cv2.imshow("IPWebcam",img)
#         if cv2.waitKey(1)==ord('q'):
#             break
#     cv2.waitKey(1)
# cv2.destroyAllWindows()


import sys
import logging as log
import datetime as dt
from time import sleep
import cv2
import os
import urllib.request
import numpy as np
import face_recognition
from datetime import datetime

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)
url="http://192.168.1.234:8080/shot.jpg" 

path = 'imgs'
images = []
className = []
myList = os.listdir(path)
for c in myList:
    newImg = cv2.imread(f'{path}/{c}')
    images.append(newImg)
    className.append(os.path.splitext(c)[0])
print(className)

def finEncode(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def addtofile(name):
    with open("attendaneName.csv", "r+") as f:
        datalist = f.readlines()
        namelist = []
        for l in datalist:
            existname = l.split(',')
            namelist.append(existname[0])
        if name not in namelist:
            now = datetime.now()
            dateformat = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateformat}')

encodeImages = finEncode(images)
print("Encoding Done")

while True:
    success, img = cv2.VideoCapture(url).read()

    if not success:
        print("Error reading frame from video stream")
        break

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceLoc = face_recognition.face_locations(imgSmall)
    faceEncode = face_recognition.face_encodings(imgSmall, faceLoc)

    for loc, encode in zip(faceLoc, faceEncode):
        match = face_recognition.compare_faces(encodeImages, encode)
        dist = face_recognition.face_distance(encodeImages, encode)
        print(dist)
        indx = np.argmin(dist)

        if match[indx]:
            name = className[indx].upper()
        else:
            name = "Unknown"

        y1, x2, y2, x1 = loc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        addtofile(name)
        cv2.imshow("IPWebcam",img)
        if cv2.waitKey(1)==ord('q'):
            break


# import sys
# import logging as log
# import datetime as dt
# from time import sleep
# import cv2
# import os
# import numpy as np
# import face_recognition
# from datetime import datetime

# cascPath = "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
# log.basicConfig(filename='webcam.log', level=log.INFO)
# url = "http://192.168.1.234:8080/shot.jpg"
# img_url=np.array(bytearray(urllib.request.urlopen(url).read()),dtype=np.uint8)
# img=cv2.imdecode(img_url,-1)

# path = 'imgs'
# images = []
# className = []
# myList = os.listdir(path)
# for c in myList:
#     newImg = cv2.imread(f'{path}/{c}')
#     images.append(newImg)
#     className.append(os.path.splitext(c)[0])
# print(className)

# def finEncode(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# def addtofile(name):
#     with open("attendaneName.csv", "r+") as f:
#         datalist = f.readlines()
#         namelist = []
#         for l in datalist:
#             existname = l.split(',')
#             namelist.append(existname[0])
#         if name not in namelist:
#             now = datetime.now()
#             dateformat = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dateformat}')

# encodeImages = finEncode(images)
# print("Encoding Done")

# cam = cv2.VideoCapture(url)

# while True:
#     success, img = cam.read()

#     if not success:
#         print("Error reading frame from video stream")
#         break

#     imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

#     faceLoc = face_recognition.face_locations(imgSmall)
#     faceEncode = face_recognition.face_encodings(imgSmall, faceLoc)

#     for loc, encode in zip(faceLoc, faceEncode):
#         match = face_recognition.compare_faces(encodeImages, encode)
#         dist = face_recognition.face_distance(encodeImages, encode)
#         print(dist)
#         indx = np.argmin(dist)

#         if match[indx]:
#             name = className[indx].upper()
#         else:
#             name = "Unknown"

#         y1, x2, y2, x1 = loc
#         y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#         cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#         addtofile(name)
#         cv2.imshow("IPWebcam",img)
#         if cv2.waitKey(1)==ord('q'):
#             break

    

# cv2.destroyAllWindows()


# When everything is done, release the capture



# path='imgs'
# images=[]
# className=[]
# myList=os.listdir(path)
# for c in myList:
#     newImg=cv2.imread(f'{path}/{c}')
#     images.append(newImg)
#     className.append(os.path.splitext(c)[0])
# print(className)
# def finEncode(images):
#     encodeList=[]
#     for img in images:
#         img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         encode=face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# encodeImages=finEncode(images)
# print("encodeing Done")

# cam=cv2.VideoCapture(0)
# while True:
#     success,img=cam.read()
#     imgSmall=cv2.resize(img,(0, 0),None,0.25,0.25)
#     imgSmall=cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

#     faceLoc=face_recognition.face_locations(imgSmall)
#     faceEncode=face_recognition.face_encodings(imgSmall,faceLoc)
#     for loc,encode in zip(faceLoc,faceEncode):
#         match=face_recognition.compare_faces(encodeImages,encode)
#         dist=face_recognition.face_distance(encodeImages,encode)
#         print(dist)
     

#     cv2.imshow("webcam",img)
#     cv2.waitKey(1)







# video_capture = cv2.VideoCapture(0)
# anterior = 0

# while True:
#     if not video_capture.isOpened():
#         print('Unable to load camera.')
#         sleep(5)
#         pass

#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )

#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     if anterior != len(faces):
#         anterior = len(faces)
#         log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


#     # Display the resulting frame
#     cv2.imshow('Video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('s'): 

#         check, frame = video_capture.read()
#         cv2.imshow("Capturing", frame)
#         cv2.imwrite(filename='savedimg.jpg', img=frame)
#         video_capture.release()
#         img_new = cv2.imread('savedimg.jpg', cv2.IMREAD_GRAYSCALE)
#         img_new = cv2.imshow("Captured Image", img_new)
#         cv2.waitKey(1650)
#         print("Image Saved")
#         print("Program End")
#         cv2.destroyAllWindows()

#         break
#     elif cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Turning off camera.")
#         video_capture.release()
#         print("Camera off.")
#         print("Program ended.")
#         cv2.destroyAllWindows()
#         break

#     # Display the resulting frame
#     cv2.imshow('Video', frame)
# video_capture.release()
# import sys
# import logging as log
# import datetime as dt
# from time import sleep
# import cv2
# import os
# import numpy as np
# import face_recognition
# from datetime import datetime 
# cascPath = "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
# log.basicConfig(filename='webcam.log',level=log.INFO)
# url="http://192.168.1.234:8080/shot.jpg"






# path='imgs'
# images=[]
# className=[]
# myList=os.listdir(path)
# for c in myList:
#     newImg=cv2.imread(f'{path}/{c}')
#     images.append(newImg)
#     className.append(os.path.splitext(c)[0])
# print(className)
# def finEncode(images):
#     encodeList=[]
#     for img in images:
#         img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         encode=face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# def addtofile(name):
#     with open("attendaneName.csv","r+")as f:
#         datalist=f.readlines()
#         namelist=[]
#         for l in datalist:
#             exsistname=l.split(',')
#             namelist.append(exsistname[0])
#         if name not in namelist:
#             now =datetime.now()
#             dateformat=now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dateformat}')

# encodeImages=finEncode(images)
# print("encodeing Done")


# cam=cv2.VideoCapture(url)
# while True:
#     success,img=cam.read()
#     imgSmall=cv2.resize(img,(0, 0),None,0.25,0.25)
#     imgSmall=cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

#     faceLoc=face_recognition.face_locations(imgSmall)
#     faceEncode=face_recognition.face_encodings(imgSmall,faceLoc)
#     for loc,encode in zip(faceLoc,faceEncode):
#         match=face_recognition.compare_faces(encodeImages,encode)
#         dist=face_recognition.face_distance(encodeImages,encode)
#         print(dist)
#         indx = np.argmin(dist)
#         if match[indx]:
#             name=className[indx].upper()
#         else:
#             name="unknown"
#         y1 , x2, y2 , x1=  loc
#         y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
#         cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#         cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#         cv2.putText(img ,name ,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
#         addtofile(name)
#     while True:
#         img_url=np.array(bytearray(urllib.request.urlopen(url).read()),dtype=np.uint8)
#         img=cv2.imdecode(img_url,-1)
#         cv2.imshow("IPWebcam",img)
#         if cv2.waitKey(1)==ord('q'):
#             break
#     cv2.waitKey(1)
# cv2.destroyAllWindows()


import sys
import logging as log
import datetime as dt
from time import sleep
import cv2
import os
import urllib.request
import numpy as np
import face_recognition
from datetime import datetime

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)
url="http://192.168.1.234:8080/shot.jpg" 

path = 'imgs'
images = []
className = []
myList = os.listdir(path)
for c in myList:
    newImg = cv2.imread(f'{path}/{c}')
    images.append(newImg)
    className.append(os.path.splitext(c)[0])
print(className)

def finEncode(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def addtofile(name):
    with open("attendaneName.csv", "r+") as f:
        datalist = f.readlines()
        namelist = []
        for l in datalist:
            existname = l.split(',')
            namelist.append(existname[0])
        if name not in namelist:
            now = datetime.now()
            dateformat = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateformat}')

encodeImages = finEncode(images)
print("Encoding Done")

while True:
    success, img = cv2.VideoCapture(url).read()

    if not success:
        print("Error reading frame from video stream")
        break

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceLoc = face_recognition.face_locations(imgSmall)
    faceEncode = face_recognition.face_encodings(imgSmall, faceLoc)

    for loc, encode in zip(faceLoc, faceEncode):
        match = face_recognition.compare_faces(encodeImages, encode)
        dist = face_recognition.face_distance(encodeImages, encode)
        print(dist)
        indx = np.argmin(dist)

        if match[indx]:
            name = className[indx].upper()
        else:
            name = "Unknown"

        y1, x2, y2, x1 = loc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        addtofile(name)
        cv2.imshow("IPWebcam",img)
        if cv2.waitKey(1)==ord('q'):
            break


# import sys
# import logging as log
# import datetime as dt
# from time import sleep
# import cv2
# import os
# import numpy as np
# import face_recognition
# from datetime import datetime

# cascPath = "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)
# log.basicConfig(filename='webcam.log', level=log.INFO)
# url = "http://192.168.1.234:8080/shot.jpg"
# img_url=np.array(bytearray(urllib.request.urlopen(url).read()),dtype=np.uint8)
# img=cv2.imdecode(img_url,-1)

# path = 'imgs'
# images = []
# className = []
# myList = os.listdir(path)
# for c in myList:
#     newImg = cv2.imread(f'{path}/{c}')
#     images.append(newImg)
#     className.append(os.path.splitext(c)[0])
# print(className)

# def finEncode(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# def addtofile(name):
#     with open("attendaneName.csv", "r+") as f:
#         datalist = f.readlines()
#         namelist = []
#         for l in datalist:
#             existname = l.split(',')
#             namelist.append(existname[0])
#         if name not in namelist:
#             now = datetime.now()
#             dateformat = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dateformat}')

# encodeImages = finEncode(images)
# print("Encoding Done")

# cam = cv2.VideoCapture(url)

# while True:
#     success, img = cam.read()

#     if not success:
#         print("Error reading frame from video stream")
#         break

#     imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

#     faceLoc = face_recognition.face_locations(imgSmall)
#     faceEncode = face_recognition.face_encodings(imgSmall, faceLoc)

#     for loc, encode in zip(faceLoc, faceEncode):
#         match = face_recognition.compare_faces(encodeImages, encode)
#         dist = face_recognition.face_distance(encodeImages, encode)
#         print(dist)
#         indx = np.argmin(dist)

#         if match[indx]:
#             name = className[indx].upper()
#         else:
#             name = "Unknown"

#         y1, x2, y2, x1 = loc
#         y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#         cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#         addtofile(name)
#         cv2.imshow("IPWebcam",img)
#         if cv2.waitKey(1)==ord('q'):
#             break

    

# cv2.destroyAllWindows()


# When everything is done, release the capture



# path='imgs'
# images=[]
# className=[]
# myList=os.listdir(path)
# for c in myList:
#     newImg=cv2.imread(f'{path}/{c}')
#     images.append(newImg)
#     className.append(os.path.splitext(c)[0])
# print(className)
# def finEncode(images):
#     encodeList=[]
#     for img in images:
#         img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         encode=face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# encodeImages=finEncode(images)
# print("encodeing Done")

# cam=cv2.VideoCapture(0)
# while True:
#     success,img=cam.read()
#     imgSmall=cv2.resize(img,(0, 0),None,0.25,0.25)
#     imgSmall=cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

#     faceLoc=face_recognition.face_locations(imgSmall)
#     faceEncode=face_recognition.face_encodings(imgSmall,faceLoc)
#     for loc,encode in zip(faceLoc,faceEncode):
#         match=face_recognition.compare_faces(encodeImages,encode)
#         dist=face_recognition.face_distance(encodeImages,encode)
#         print(dist)
     

#     cv2.imshow("webcam",img)
#     cv2.waitKey(1)







# video_capture = cv2.VideoCapture(0)
# anterior = 0

# while True:
#     if not video_capture.isOpened():
#         print('Unable to load camera.')
#         sleep(5)
#         pass

#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )

#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     if anterior != len(faces):
#         anterior = len(faces)
#         log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


#     # Display the resulting frame
#     cv2.imshow('Video', frame)

#     if cv2.waitKey(1) & 0xFF == ord('s'): 

#         check, frame = video_capture.read()
#         cv2.imshow("Capturing", frame)
#         cv2.imwrite(filename='savedimg.jpg', img=frame)
#         video_capture.release()
#         img_new = cv2.imread('savedimg.jpg', cv2.IMREAD_GRAYSCALE)
#         img_new = cv2.imshow("Captured Image", img_new)
#         cv2.waitKey(1650)
#         print("Image Saved")
#         print("Program End")
#         cv2.destroyAllWindows()

#         break
#     elif cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Turning off camera.")
#         video_capture.release()
#         print("Camera off.")
#         print("Program ended.")
#         cv2.destroyAllWindows()
#         break

#     # Display the resulting frame
#     cv2.imshow('Video', frame)
# video_capture.release()