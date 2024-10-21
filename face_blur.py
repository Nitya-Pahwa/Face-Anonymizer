import cv2 as cv
import mediapipe as mp
# to read the image
path='lady.jpg'
img=cv.imread(path)
H,W,_=img.shape
# detect faces
face=mp.solutions.face_detection
with face.FaceDetection(model_selection=0,min_detection_confidence=0.5) as detect:
    rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    out=detect.process(rgb)
    if out.detections is not None:
        for detection in out.detections:
            ld=detection.location_data
            box=ld.relative_bounding_box
            x,y,h,w=box.xmin,box.ymin,box.height,box.width

            x=int(x*W)
            y=int(y*H)
            w=int(w*W)
            h=int(h*H)
        # blur faces
        img[y:y+h,x:x+w,:]=cv.blur(img[y:y+h,x:x+w,:],(50,50))

    cv.imshow('Image',img)
    cv.waitKey(0)