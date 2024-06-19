import cv2
import numpy as np
import os


def severity_param(img):
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([20, 255, 200])


    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    brown_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)


    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_brown, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    inten_yellow = []
    inten_brown = []

    for cnt in contours_yellow:
        mask = np.zeros_like(yellow_mask)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        intensity = cv2.mean(hsv_img, mask=mask)[2]
        inten_yellow = np.append(inten_yellow,intensity)
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)


        
    for cnt in contours_brown:
        mask = np.zeros_like(brown_mask)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        intensity = cv2.mean(hsv_img, mask=mask)[2]
        inten_brown = np.append(inten_brown,intensity)
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2)
    
        
    area_damaged1 = 0
    area_damaged2 = 0
    
    for cnt in contours_yellow:
        area_damaged1 += cv2.contourArea(cnt)
        
    for cnt in contours_brown:
        area_damaged2 += cv2.contourArea(cnt)
        
    area_damaged = (area_damaged1 + area_damaged2)
    
    try:
    
        if np.max(inten_yellow) == 0 and area_damaged == 0:
            return 'Healthy'
    
        if np.max(inten_yellow) < 65:
            return '0'
        
        if 65 < np.max(inten_yellow) < 130:
            return '1'
            
        if 130 < np.max(inten_yellow) < 195:
            return '2'
    
        if 195 < np.max(inten_yellow) < 260:
            return '3'
        
    except:
        pass



def save_segmented_leaves_from_video(save_directory, video_path, fps):

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
        
        try:
            
            cv2.drawContours(mask, [contours[-1]], -1, (255, 255, 255), thickness=-1)
            
        except:
            
            pass

        result = cv2.bitwise_and(frame, mask)
        
        image_path = os.path.join(save_directory, f"segmented_leaf_{frame_count}.jpg")
        cv2.imwrite(image_path, result)
        image_paths.append(image_path)

        severity = severity_param(result)

        print(severity)

        frame_count += 1
        

    cap.release()

image = save_segmented_leaves_from_video('D:/extra_data/segmented','D:/extra_data/leaves-25fps.mp4', 25)