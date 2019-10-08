
# coding: utf-8

# Pre processing data to store numpy array of the image

# In[ ]:


X = [] #skin cancer image
y = [] #mask of skin cancer
IMG_SIZE=256

def create_train_data():
    
    height = 256
    width = 256
    DATADIR = "C:/Users/Ruchi Awasthi/Downloads/Data/Train/Skin" #local directory for skin cancer original images
    DATADIR2 = "C:/Users/Ruchi Awasthi/Downloads/Data/Train/Mask" #local directory for skin cancer mask images

    for img in os.listdir(DATADIR):
        try:
            img_arr=cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE)
            img_arr = cv2.resize(img_arr, (width, height))
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            
            #gaussian blur
            img_arr = ndimage.gaussian_filter(img_arr, sigma=1)
            
            #contrast stretching
            f_max = (np.amax(img_arr))
            f_min = (np.amin(img_arr))
            img_arr = ((img_arr - f_min)/(f_max - f_min))*255
            
            #sharpen
            filter_blurred_f = ndimage.gaussian_filter(img_arr, sigma=2)
            alpha = 10
            img_arr = img_arr + alpha * (img_arr -filter_blurred_f)
            
            
            new_arr = img_arr/255.0
            X.append(new_arr)
            
        except Exception as e:
            pass
        
    for img in os.listdir(DATADIR2):
        try:        
            truth_img_arr=cv2.imread(os.path.join(DATADIR2, img), cv2.IMREAD_GRAYSCALE)
            truth_new_arr = cv2.resize(truth_img_arr, (height, width))
            truth_new_arr = truth_new_arr/255.0
            y.append(truth_new_arr)
        except Exception as e:
            pass
create_train_data()

print(len(X))
print(len(y))

height = 256
width = 256
X = np.array(X).reshape(-1, width, height, 1)
y = np.array(y).reshape(-1, width, height, 1)
print(X.shape)
print(y.shape)

np.save("C:/Users/Ruchi Awasthi/Downloads/Data/Train_Skin.npy",X)
np.save("C:/Users/Ruchi Awasthi/Downloads/Data/Train_Mask.npy",y)


X = [] #skin cancer image
y = [] #mask of skin cancer
IMG_SIZE=256

def create_test_data():
    
    height = 256
    width = 256
    DATADIR = "C:/Users/Ruchi Awasthi/Downloads/Data/Test/Skin" #local directory for skin cancer original images
    DATADIR2 = "C:/Users/Ruchi Awasthi/Downloads/Data/Test/Mask" #local directory for skin cancer mask images

    for img in os.listdir(DATADIR):
        try:
            img_arr=cv2.imread(os.path.join(DATADIR, img), cv2.IMREAD_GRAYSCALE)
            img_arr = cv2.resize(img_arr, (width, height))
            
            #gaussian blur
            img_arr = ndimage.gaussian_filter(img_arr, sigma=1)
            
            #contrast stretching
            f_max = (np.amax(img_arr))
            f_min = (np.amin(img_arr))
            img_arr = ((img_arr - f_min)/(f_max - f_min))*255
            
            #sharpen
            filter_blurred_f = ndimage.gaussian_filter(img_arr, 2)
            alpha = 10
            img_arr = img_arr + alpha * (img_arr -filter_blurred_f)
            
            
            new_arr = img_arr/255.0
            X.append(new_arr)
            
        except Exception as e:
            pass
        
    for img in os.listdir(DATADIR2):
        try:        
            truth_img_arr=cv2.imread(os.path.join(DATADIR2, img), cv2.IMREAD_GRAYSCALE)
            truth_new_arr = cv2.resize(truth_img_arr, (height, width))
            truth_new_arr = truth_new_arr/255.0
            y.append(truth_new_arr)
        except Exception as e:
            pass
create_test_data()

print(len(X))
print(len(y))

height = 256
width = 256
X = np.array(X).reshape(-1, width, height, 1)
y = np.array(y).reshape(-1, width, height, 1)
print(X.shape)
print(y.shape)

np.save("C:/Users/Ruchi Awasthi/Downloads/Data/Test_Skin.npy",X)
np.save("C:/Users/Ruchi Awasthi/Downloads/Data/Test_Mask.npy",y)


# In[ ]:


X = np.load("C:/Users/Ruchi Awasthi/Downloads/Data/Train_Skin.npy")
y = np.load("C:/Users/Ruchi Awasthi/Downloads/Data/Train_Mask.npy")

