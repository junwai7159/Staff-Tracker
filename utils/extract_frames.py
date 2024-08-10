import os 
import cv2

video_path = './data/sample.mp4'
output_dir = './imgs/imgs_raw'

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
  print('Error: Could not open video.')
  exit()

frame_count = 0

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # Save the frame as an image file
  frame_dir = os.path.join(output_dir, f'{frame_count:04d}.jpg')
  cv2.imwrite(frame_dir, frame)

  frame_count += 1

cap.release()

print(f'Extracted {frame_count} frames to {output_dir}')