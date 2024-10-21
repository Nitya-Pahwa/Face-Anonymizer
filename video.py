import cv2 as cv
import mediapipe as mp
import argparse
def process_img(img,detect):
    rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    out=detect.process(rgb)
    H,W,_=img.shape
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
    return img

args=argparse.ArgumentParser()
args.add_argument("--mode",default='webcam')
args.add_argument("--filePath",default='None')
args=args.parse_args()
# detect faces
face=mp.solutions.face_detection
with face.FaceDetection(model_selection=0,min_detection_confidence=0.5) as detect:
    if args.mode in ["image"]:
        img=cv.imread(args.filePath)
        
        img=process_img(img,detect)
        cv.imshow('Image',img)
        cv.waitKey(0)
    elif args.mode in ["video"]:
        cap=cv.VideoCapture(args.filePath)
        ret,frame=cap.read()
        while ret:
            frame=process_img(frame,detect)
            ret,frame=cap.read()
            cv.imshow('Frame',frame)
            if(cv.waitKey(1) & 0xFF==ord('q')):
                break
        cap.release()
    elif args.mode in ["webcam"]:
        cap=cv.VideoCapture(0)
        ret,frame=cap.read()
        while ret:
            frame=process_img(frame,detect)
            
            cv.imshow('Frame',frame)
            if(cv.waitKey(1) & 0xFF==ord('q')):
                break
            ret,frame=cap.read()
        cap.release()