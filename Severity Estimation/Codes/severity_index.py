import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


files = glob.glob('D:/Sintu/tomato_leaf_detection/train/Tomato___Bacterial_spot/*.jpg')


def severity_param(img_path):
    
    img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    


    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([20, 255, 200])


    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    brown_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_brown, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    leaf_contour = contours[0]
  
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
    leaf_area = cv2.contourArea(leaf_contour)
    
    return leaf_area, area_damaged, inten_brown, inten_yellow



# Finding Parameters Threshold values for Labelling Severity Levels

leaf_area = []
area_damaged = []
intensity_yellow = []
intensity_brown = []

for i in range(len(files)):
    severity = severity_param(files[i])
    leaf_area = np.append(leaf_area, severity[0])
    area_damaged = np.append(area_damaged, severity[1])
    if len(severity[2]) == 0:
        intensity_brown = np.append(intensity_brown, 0)
    if len(severity[2]) != 0:
        intensity_brown = np.append(intensity_brown, np.max(severity[2]))
    if len(severity[3]) == 0:
        intensity_yellow = np.append(intensity_yellow, 0)
    if len(severity[3]) != 0:
        intensity_yellow = np.append(intensity_yellow, np.max(severity[3]))



severity_index = []        
        
for i in range(len(files)):
    w1 = 0.17
    w2 = 0.05
    w3 = 0.78
    damaged = area_damaged[i]
    inten_brown = intensity_brown[i]
    inten_yellow = intensity_yellow[i]
    
    if damaged == 0 and inten_brown == 0 and inten_yellow == 0:
        severity_index = np.append(severity_index, 0)
        
    else: 
        severity_index = np.append(severity_index, (w1*damaged + w2*inten_brown + w3*inten_yellow)/(damaged + inten_brown + inten_yellow))
    
severity = []

for i in range(len(files)):
    
    if severity_index[i] == 0:
        severity = np.append(severity, 0)
    
    if 0 < severity_index[i] <= 0.19:
        severity = np.append(severity, 1)
        
    if 0.19 < severity_index[i] <= 0.39:
        severity = np.append(severity, 2)
        
    if 0.39 < severity_index[i] <= 0.59:
        severity = np.append(severity, 3)
        
    if 0.59 < severity_index[i] <= 0.79:
        severity = np.append(severity, 4)
         
    if 0.79 < severity_index[i] <= 1:
        severity = np.append(severity, 5)
        
        
severity = severity.astype(int)
        
value_counts = np.bincount(severity)

x = np.arange(len(value_counts))

plt.bar(x, value_counts)

plt.xticks(x)

plt.xlabel("No. of Samples")
plt.ylabel("Severity")

plt.title("Severity using Image Processing")

plt.show()