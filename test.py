
# coding: utf-8

# In[ ]:


X_test=[]
y_test=[]
X_test = np.load("C:/Users/Kritagya Nayyar/Desktop/Skin Cancer Paper/Test_Skin.npy")   #running this model we will generate sgemented images i.e. eval_segm
y_mask = np.load("C:/Users/Kritagya Nayyar/Desktop/Skin Cancer Paper/Test_Mask.npy")   #GT of segmented test images i.e. gt_segm

print(X_test.shape)
print(y_mask.shape)


# In[ ]:


# model = unet()
# model.load_weights("C:/Users/Kritagya Nayyar/Desktop/Skin Cancer Paper/model_training_file.hdf5")
eval_segm = model.predict(X_test,batch_size=16,verbose=1)

