import cv2
import numpy as np
from utils import *

# Connect to a live stream
stream_url = "rtmp://192.168.25.1:8082/live"
model = load_maskrcnn()

# Create an OpenCV video capture object to decode the frames from the stream
#cap = cv2.VideoCapture(stream_url)
cap = cv2.VideoCapture("vidroom.mp4")

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

# Loop over the frames in the stream, process them, and write them to the output video
while cap.isOpened():
    # Read the next frame from the stream
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame with PyTorch
    result = feedforward(model, frame)
    result_frame = PIL_to_cv2(result)
    # Write the processed frame to the output video
    out.write(result_frame)
    
    # Send the processed frame to the VLC stream
    #cv2.imshow("Processed Frame", frame)
    #if cv2.waitKey(1) & 0xFF == ord("q"):
    #    break
    #cv2.waitKey(1)

# Release the resources
out.release()
cap.release()
cv2.destroyAllWindows()

