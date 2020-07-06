from facedetector import FaceDetector
import imutils
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True, help = "path to where the face cascade resides")
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
args = vars(ap.parse_args())

fd = FaceDetector(args["face"])

if not args.get("video", False):
    camera = cv2.VideoCapture()
    camera.open("rtsp://<username>:<password>@<ipaddress>:554/Streaming/channels/201")

else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read() #Grabbed: Successful reading of frame

    if args.get("video") and not grabbed:
        break

    #frame = imutils.resize(frame, width = 300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = fd.detect(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))

    frameClone = frame.copy()

    for (fX, fY, fW, fH) in faceRects:
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 255, 0), 2)

    cv2.imshow("Face", frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()