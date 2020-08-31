import numpy as np
import cv2
import random
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else 'Animation1.mp4'
video = cv2.VideoCapture(filename)
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
hasFrame, frame0 = video.read()
frame = cv2.resize(frame0,(frame0.shape[1]//2,frame0.shape[0]//2))

refPt0 = refPt1 = None
selPt0 = selPt1 = None
def mouseHandler(event, x, y, flags, param):
    global refPt0, refPt1, selPt0, selPt1
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt0 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selPt0 = refPt0
        selPt1 = (x,y) 
        refPt0 = refPt1 = None
    elif refPt0 is not None:
        refPt1 = (x,y)
        
cv2.namedWindow("crop video")
cv2.setMouseCallback("crop video", mouseHandler)
cv2.imshow("crop video", frame)
cv2.waitKey(1)

while True:
    res = np.copy(frame)
    if selPt0 is not None and selPt1 is not None:
        tl = (min(selPt0[0],selPt1[0]),min(selPt0[1],selPt1[1]))
        br = (max(selPt0[0],selPt1[0]),max(selPt0[1],selPt1[1]))
        print(tl,br)
        selPt0 = setPt1 = None
        break
    elif refPt0 is not None and refPt1 is not None:
        cv2.rectangle(res,(min(refPt0[0],refPt1[0]),min(refPt0[1],refPt1[1])),(max(refPt0[0],refPt1[0]),max(refPt0[1],refPt1[1])),(0,255,0),1)
    
    cv2.imshow("crop video", res)
    cv2.waitKey(1)

tl = tuple(2*i for i in tl)
br = tuple(2*i for i in br)
frame = frame0

out = cv2.VideoWriter()
out.open(filename[:-4]+'-cropped.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,(frame.shape[1],frame.shape[0]))

while True:

    target = frame[tl[1]:br[1],tl[0]:br[0],:]
    coef = min(frame.shape[1]/target.shape[1],frame.shape[0]/target.shape[0])
    target = cv2.resize(target,(int(target.shape[1]*coef),int(target.shape[0]*coef)))
    result = np.zeros(frame.shape,np.uint8)
    p = ((frame.shape[1]-target.shape[1])//2,(frame.shape[0]-target.shape[0])//2)
    result[p[1]:p[1]+target.shape[0],p[0]:p[0]+target.shape[1],:] = target

    out.write(result)

    cv2.imshow("crop video", result)
    cv2.waitKey(1)
    
    hasFrame, frame = video.read()
    if not hasFrame:
        break

out.release()
cv2.destroyAllWindows()
