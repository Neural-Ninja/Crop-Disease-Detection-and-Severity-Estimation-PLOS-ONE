import cv2
import numpy as np
import os

def save_segmented_leaves_from_video(video_path, fps, save_directory):

    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():
        print("Failed to open the video file")
        return []

    os.makedirs(save_directory, exist_ok=True)


    image_paths = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_img_leaf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_img_leaf, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=3)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(frame)
        cv2.drawContours(mask, [contours[-1]], -1, (255, 255, 255), thickness=-1)

        result = cv2.bitwise_and(frame, mask)

        image_path = os.path.join(save_directory, f"segmented_leaf_{frame_count}.jpg")
        cv2.imwrite(image_path, result)
        image_paths.append(image_path)

        frame_count += 1

    cap.release()

    return image_paths


image = save_segmented_leaves_from_video('D:/extra_data/leaves-25fps.mp4', 25, 'D:/extra_data/segmented')