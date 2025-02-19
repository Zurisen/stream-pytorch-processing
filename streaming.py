import cv2
import numpy as np
from helpers import *

# Connect to a live stream
stream_url = "rtmp://192.168.25.1:8082/live"

# Create an OpenCV video capture object to decode the frames from the stream
cap = cv2.VideoCapture(stream_url)
#cap = cv2.VideoCapture("etc/vidroom2.mp4")

# Define the codec and output format for the processed video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

# Create a VLC HTTP or RTP stream for the output video
# Example for RTP:
#vlc_url = "rtp://127.0.0.1:1234/out_stream"
# Example for HTTP:
# vlc_url = "http://127.0.0.1:8080/"


cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
# Loop over the frames in the stream, process them, and write them to the output video
while cap.isOpened():
    # Read the next frame from the stream
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)
    if not ret:
        break
    if frame is None:
        print('Finished!')
        out.release()
        break
    counter += 1
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 0)
    detections = model(frame)

    if detections is not None:
        print("N det: ", len(detections.pandas().xyxy[0]))
        tracked_objects = mot_tracker.update(detections.pandas().xyxy[0].iloc[:,:5].to_numpy(), detections.pandas().xyxy[0].name.values)
        #tracked_objects = mot_tracker.update(detections.pandas().xyxy[0].to_numpy())
        print("N tracked: ", len(tracked_objects))

        for x1, y1, x2, y2, obj_id, obj_class in tracked_objects:
            x1, y1, x2, y2, obj_id = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2)), int(obj_id)

            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            #cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            #cv2.rectangle(frame, (x1, y1-35), (x1+len(str(obj_id))*19+60, y1), color, -1)
            #cv2.putText(frame, str(obj_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            
            # Draw the bounding box around the tracked object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame, (x1, y1), (x1+len(obj_class+' - '+str(obj_id))*18, y1-25), color, -1)
            cv2.putText(frame, obj_class+' - '+str(obj_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    out.write(frame)
    clear_output(wait=True)

# Release the resources
out.release()
cap.release()
cv2.destroyAllWindows()

