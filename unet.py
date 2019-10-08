
# coding: utf-8

# In[ ]:


def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    
    
    G=32
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    #3x3 matric on i'inputs' - 2 times
    bn1=GroupNormalization(groups=G)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    #Pooling - 2x2 matrix applied on conv1

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    #3x3 matric on pooled data 
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    bn2=GroupNormalization(groups=G)(conv2)
    #3x3 matrix on convolution applied on pooled data
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    #Pooling - 2x2 matrix applied on conv2
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    #3x3 matric on pooled data
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    bn3=GroupNormalization(groups=G)(conv3)
    #3x3 matrix on convolution applied on pooled data
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
    #Pooling - 2x2 matrix applied on conv3

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    bn4=GroupNormalization(groups=G)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)
    
    

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=1)(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=4)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=8)(conv5)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=16)(conv5)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=32)(conv5)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    #image converted to 512 pixels
    #bottleneck
#     dilate1 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=1, kernel_initializer='he_normal')(down3pool)
    
    
    
    bn5=GroupNormalization(groups=G)(conv5)
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn5), bn4], axis=3)
    #up convulation aaplied on conv5 (512 pixels) which is to be superimposed with conv 4 (256 pixels) using 2x2 matrix
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    #3x3 matrix on up6 - convert to 256 pixels
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    bn6=GroupNormalization(groups=G)(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn6), bn3], axis=3)
    #up convulation aaplied on conv6 (256 pixels) which is to be superimposed with conv3 (128 pixels) 
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    #3x3 matrix on up6 - convert to 128 pixels
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    bn7=GroupNormalization(groups=G)(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn7), bn2], axis=3)
    #up convulation aaplied on conv7 (128 pixels) which is to be superimposed with conv2 (64 pixels)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    #3x3 matrix on up6 - convert to 64 pixels
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    bn8=GroupNormalization(groups=G)(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(bn8), bn1], axis=3)
    #up convulation aaplied on conv8 (64 pixels) which is to be superimposed with conv1 (32 pixels)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    #3x3 matrix on up6 - convert to 32 pixels
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    bn9=GroupNormalization(groups=G)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)
    #conv9 converted to the image of original size under sigmoid activation by 1x1 matrix. The matrix is named conv10

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = generalized_dice_loss, metrics = ['accuracy', jaccard_distance, sensitivity, specificity])
#     model.compile(optimizer = Adagrad(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'hinge', metrics = ['accuracy'])
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'squared_hinge', metrics = ['accuracy'])
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'logcosh', metrics = ['accuracy'])
    
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'kullback_leibler_divergence', metrics = ['accuracy'])
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'poisson', metrics = ['accuracy'])
    
    
    
    return model

