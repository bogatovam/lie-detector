import cv2
import numpy as np
from tqdm import tqdm

VIDEO_FILE_NAME = ''
cap = cv2.VideoCapture(VIDEO_FILE_NAME)
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

def select_face(frame, faces):
    # increase face window
    x = faces[0][0] - 1 * faces[0][0] // 10
    y = faces[0][1] - 1 * faces[0][1] // 10
    w = faces[0][2] + 2 * faces[0][2] // 10
    h = faces[0][3] + 2 * faces[0][3] // 10

    pts = np.array([[0,0],[width,0],[width,y],[0,y]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(frame,[pts],(0,0,0),4)
                
    pts = np.array([[0,y],[x,y],[x,y + h],[0,y + h]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(frame,[pts],(0,0,0),4)

    pts = np.array([[x + w,y],[width,y],[width,y+h],[x + w,y+h]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(frame,[pts],(0,0,0),4)

    pts = np.array([[0,y+h],[width,y+h],[width,height],[0,height]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(frame,[pts],(0,0,0),4)    

frames = []
o_f_c = [] #old face cords
size = None
print('processing {}'.format(VIDEO_FILE_NAME))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in tqdm(range(length)):
    ret,frame = cap.read()
    height, width, layers = frame.shape
    size = (width,height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) != 0:
        select_face(frame, faces)
        o_f_c = [faces[0][0], faces[0][1], faces[0][2], faces[0][3]]
    else:
        if len(o_f_c) != 0:
            select_face(frame, [o_f_c])
        else:
            pts = np.array([[0,0],[width,0],[width,height],[0,height]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(frame,[pts],(0,0,0),4)
    frames.append(frame)
#    cv2.imshow('window-name', frame)
#    if cv2.waitKey(5) & 0xFF == ord('q'):
#        break    

print('saving project.avi')
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for frame in frames:
    out.write(frame)
out.release()
cap.release()
cv2.destroyAllWindows() # destroy all opened windows
