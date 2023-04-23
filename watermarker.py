import numpy as np
import cv2
import random
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else 'Animation1.mp4'
watermark = sys.argv[2] if len(sys.argv) > 2 else 'watermark'
video = cv2.VideoCapture(filename)
fps = video.get(cv2.CAP_PROP_FPS)
print('fps',fps)
hasFrame, frame0 = video.read()
frame = cv2.resize(frame0,(frame0.shape[1]//2,frame0.shape[0]//2))

def split_text(text, max_length):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + word) + 1 <= max_length:
            # Add the word to the current line
            if current_line:
                current_line += " "
            current_line += word
        else:
            # Start a new line with the current word
            lines.append(current_line)
            current_line = word
    # Add the last line
    if current_line:
        lines.append(current_line)
    return lines

def putParagraph(img, text, rect, font_scale=1, thickness=2):
    (dx, dy), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    dx /= len(text)
    spacing = int(dy * 0.4)
    x, y, w, h = rect
    n = w // dx
    lines = split_text(text,n)
    for line in lines:
        y += dy
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
        y += spacing

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
        
cv2.namedWindow("put watermark")
cv2.setMouseCallback("put watermark", mouseHandler)
cv2.imshow("put watermark", frame)
cv2.waitKey(1)

while True:
    res = np.copy(frame)
    if selPt0 is not None and selPt1 is not None:
        tl = (min(selPt0[0],selPt1[0]),min(selPt0[1],selPt1[1]))
        br = (max(selPt0[0],selPt1[0]),max(selPt0[1],selPt1[1]))
        rect = (2*tl[0],2*tl[1],2*(br[0]-tl[0]+1),2*(br[1]-tl[1]+1))
        selPt0 = setPt1 = None
        break
    elif refPt0 is not None and refPt1 is not None:
        tl = (min(refPt0[0],refPt1[0]),min(refPt0[1],refPt1[1]))
        br = (max(refPt0[0],refPt1[0]),max(refPt0[1],refPt1[1]))
        rect = (tl[0],tl[1],br[0]-tl[0]+1,br[1]-tl[1]+1)
        cv2.rectangle(res,rect,(0,255,0),1)
        putParagraph(res, watermark, rect)

    cv2.imshow("put watermark", res)
    cv2.waitKey(1)

tl = tuple(2*i for i in tl)
br = tuple(2*i for i in br)
frame = frame0

out = cv2.VideoWriter()
out.open(filename[:-4]+'-watermark.avi',cv2.VideoWriter_fourcc('M','J','P','G'),fps,(frame.shape[1],frame.shape[0]))

while True:

    result = np.copy(frame)
    putParagraph(result, watermark, rect, font_scale=2, thickness=3)

    out.write(result)

    cv2.imshow("put watermark", result)
    cv2.waitKey(1)
    
    hasFrame, frame = video.read()
    if not hasFrame:
        break

out.release()
cv2.destroyAllWindows()
