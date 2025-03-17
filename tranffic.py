from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # You can also use "yolov8s.pt" or "yolov8m.pt" for better accuracy

# Load video file instead of webcam
video_path = "27260-362770008_small.mp4"  # Change to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties for saving the output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save output video
output_path = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Detect objects
    frame = results[0].plot()  # Draw bounding boxes

    out.write(frame)  # Save frame to output video
    cv2.imshow("Traffic Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved as", output_path)
