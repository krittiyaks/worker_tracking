import cv2

# initialize the HOG descriptor/person detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("/content/object_tracking/input.mp4")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/object_tracking/output.mp4', fourcc, 20.0, (640,  480))

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      print("This Video has ended.")
      break
   
    mask = object_detector.apply(frame)

    # start detection the worker
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(mask, winStride = (8, 8), padding=(4, 4) )
    person = 1

    for i, (x,y,w,h) in enumerate(bounding_box_cordinates):
        if weights[i] < 0.60:
          continue
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
        cv2.putText(frame, f'worker:{person} acc :{weights[i]}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        person += 1
    # end detection the worker

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
