
# coding: utf-8

# In[ ]:


model = unet()
model_checkpoint = ModelCheckpoint('model_training_file.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
model.fit(X, y, batch_size=16, epochs=200, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

