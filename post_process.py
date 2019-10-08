
# coding: utf-8

# In[ ]:


#post processing

gt_segm = y_mask

gt_segm[gt_segm > .5] = 1
gt_segm[gt_segm <= 0.5] = 0

list_pred =[]
for i in range(0,len(eval_segm)):
    arr=eval_segm[i]
    arr=arr.reshape(256,256)
    arr=cv2.GaussianBlur(arr,(11,11),11)
    arr=cv2.medianBlur(arr,5)
    list_pred.append(arr)

eval_array = np.asarray(list_pred)

list_pred1 =[]
for i in range(0,len(gt_segm)):
    arr1=gt_segm[i]
    arr1=arr1.reshape(256,256)
    list_pred1.append(arr1)

gt_array = np.asarray(list_pred1)

FP=0
TP=0
FN=0
TN=0
for i in range(0,len(eval_array)):
    eval_array[i][eval_array[i] <= 0.6] = 0
    eval_array[i][eval_array[i] > 0.6] = 1
    mask_segm =(gt_array[i] - eval_array[i])
    FP = FP + np.count_nonzero(mask_segm == -1)
    TP = TP + np.count_nonzero(gt_array[i] == 1) - np.count_nonzero(mask_segm == -1)
    FN = FN + np.count_nonzero(mask_segm == 1)
    TN = TN + np.count_nonzero(mask_segm == 0) - np.count_nonzero(gt_array[i] == 1) + np.count_nonzero(mask_segm == -1)


# In[ ]:


DICE =  (2*TP)/(2*TP+FP+FN)
print (DICE)

PIXEL_ACCURACY = (TP+TN)/(TP+TN+FP+FN)
print (PIXEL_ACCURACY)

JAC = (TP)/(TP+FP+FN)
print (JAC)

Sensitivity = TP/(TP+FN)
print(Sensitivity)

Specificity = TN/(FP+TN)
print(Specificity)

Precision = TP/(TP+FP)
print(Precision)

