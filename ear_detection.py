import cv2, sys
import os
import numpy as np

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))


# left_ear_cascade = cv2.CascadeClassifier('./classifier/haarcascade_left_ear_jurij.xml')
# left_ear_cascade = cv2.CascadeClassifier('./classifier/haarcascade_right_ear_jurij.xml')

# if left_ear_cascade.empty():
  # raise IOError('Unable to load the left ear cascade classifier xml file')

# if right_ear_cascade.empty():
  # raise IOError('Unable to load the right ear cascade classifier xml file')

ear_cascade = cv2.CascadeClassifier('./classifier/vj_cascade.xml')

"""
def detectEar(img):
    detectionList = ear_cascade.detectMultiScale(img, 1.015, 8)
    return detectionList


def vizualization(img, detectionList):
    for x, y, w, h in detectionList:
        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.imwrite(filename_out + '.detected.jpg', img)
"""

TP_list = []
FP_list = []
TN_list = []
FN_list = []
F1_list = []
precision_list = []
recall_list = [] # = sensitivity


for i in range(1,251):
    filename_in = './test/' + str(i).zfill(4) + '.png'
    filename_out = './test_Results/res' + str(i).zfill(4) + '.png'
    filename_in_mask = './testannot_rect/' + str(i).zfill(4) + '.png'

    # get image and good mask:
    img = cv2.imread(filename_in)

    good_mask = cv2.imread(filename_in_mask)
    good_mask = np.array( good_mask[:,:,0], dtype=bool )

    # segment the image and get experimental mask:
    exp_mask = np.zeros( good_mask.shape, dtype= "uint8")

    detectionList = ear_cascade.detectMultiScale(img, 1.01611, 12)
    for x, y, w, h in detectionList:
        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
        cv2.rectangle(exp_mask, (x, y), (x + w, y + h), (255), -1)

    TP = np.sum(np.logical_and(good_mask, exp_mask))
    TN = np.sum(np.logical_not(np.logical_or(good_mask, exp_mask)))
    # print(TP)
    # print(TN)
    FP = np.sum(np.logical_and(np.logical_not(good_mask), exp_mask))
    FN = np.sum(np.logical_and(good_mask, np.logical_not(exp_mask)))
    # print(FP)
    # print(FN)
    #print(TP + TN + FP + FN)
    # print(img.shape[0]*img.shape[1])
    precision = TP/(TP+FP + 0.00000000001)
    recall = TP/(TP+FN + 0.00000000001)
    F1 = TP/(TP+1/2*(FP+FN) + 0.00000000001)
    #print(F1)
    TP_list.append(TP)
    FP_list.append(FP)
    TN_list.append(TN)
    FN_list.append(FN)
    F1_list.append(F1)
    precision_list.append(precision)
    recall_list.append(recall)


    cv2.imwrite(filename_out, img)

TP_array = np.array(TP_list)
FP_array = np.array(FP_list)
TN_array = np.array(TN_list)
FN_array = np.array(FN_list)
F1_array = np.array(F1_list)
precision_array = np.array(precision_list)
recall_array = np.array(recall_list)
print("TP_mean:", np.mean( TP_array ))
print("FP_mean:", np.mean( FP_array ))
print("TN_mean:", np.mean( TN_array ))
print("FN_mean:", np.mean( FN_array ))
print("F1_mean:", np.mean( F1_array ))
print("precision_mean:", np.mean( precision_array ))
print("recall_mean:", np.mean( recall_array ))