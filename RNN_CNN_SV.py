#!/usr/bin/env python
# coding: utf-8

# In[10]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys
from keras.applications.efficientnet import EfficientNetB0 , EfficientNetB4
from keras.models import Model, load_model
from keras.layers import *
from keras.layers.pooling import GlobalAveragePooling2D
#from keras.layers import LSTM , GRU, Bidirectional, LayerNormalization
#from keras.layers import TimeDistributed
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
sys.path.append("/home/gadmin/Desktop/farshid/Cyclical_Learning_Rate_main")
from clr_callback_TF2 import *
from sklearn.model_selection import LeaveOneOut,KFold,learning_curve,cross_val_score
from scikeras.wrappers import KerasRegressor
from keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
np.random.seed(0)
import pandas as pd
from keras import initializers
import keras
from keras_self_attention import SeqSelfAttention
import hydroeval as he

#num_video = 299 
index_start_event = 0
index_end_event = 145
downsamplerate = 6
bias_label = 0
height = 540
width = 380
MAX_SEQ_LENGTH=int(60/downsamplerate)
#print(MAX_SEQ_LENGTH)
size_frame=[MAX_SEQ_LENGTH,width,height,3]
#size_frame=[MAX_SEQ_LENGTH,540,960,3]
video = Input(shape=size_frame)
#print(size_frame[1:4])


# In[11]:


wp="/home/gadmin/Desktop/farshid/efficientnetb0/efficientnetb0_notop.h5"
base_feature_extractor = EfficientNetB0 (
        weights=wp,
        include_top=False,
        pooling="avg",
        input_shape=size_frame[1:4],
    )
base_feature_extractor.trainable = False
inputs = Input(size_frame[1:4])
#preprocessed = preprocess_input(inputs)
preprocessed = inputs

#print(inputs.dtype)
#print(preprocessed.dtype)

outputs1 = base_feature_extractor(preprocessed)
outputs=Dense(32,name='outputLayer')(outputs1)
feature_extractor=Model(preprocessed, outputs, name="feature_extractor")

#print(outputs.shape)


# In[12]:


# construct predictor
encoded_frames = TimeDistributed(feature_extractor)(video)
encoded_frames1 = tf.keras.layers.LayerNormalization()(encoded_frames)
encoded_sequence = LSTM(32,dropout=0.0)(encoded_frames1)
encoded_sequence1 = tf.keras.layers.LayerNormalization()(encoded_sequence)
hidden_layer = Dense(16, activation="relu")(encoded_sequence)
hidden_layer1 = Dense(8, activation="relu")(hidden_layer)
outputs = Dense(1, activation="linear")(hidden_layer1)
model_pred_RNN = Model(video, outputs)
optimizer1 = Nadam()
optimizer2 = SGD()  
#clr1 = CyclicLR(base_lr=0.00005, max_lr=0.001,
#               step_size=1, mode='triangular',scale_mode='cycle') 
optimizer3 = Adam() 
optimizer4 = RMSprop() 
#clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
#clr2 = CyclicLR(base_lr=0.001, max_lr=0.006,
#               step_size=2000, scale_fn=clr_fn,
#               scale_mode='cycle')               
                            
model_pred_RNN.compile(loss="mean_absolute_error",
              optimizer=optimizer3,
              ) 

model_pred_RNN.summary()


# In[13]:


# read event names and labels
rootpath = '/home/gadmin/Desktop/farshid/savic/convert/'
read_file = pd.read_excel("/home/gadmin/Desktop/farshid/savic/rain_estimation_savic.xlsx")
x_name=read_file.get(read_file.columns[0])
labels = read_file.get(read_file.columns[2]).values

x_name = x_name[index_start_event:index_end_event]
#print(x_name)                                           
labels=labels[index_start_event:index_end_event]*60
labels=labels+bias_label
#print(labels)
num_samples =[len(x_name)]
#print(num_samples)


# In[14]:


#print(num_samples + size_frame)


# In[15]:


# read input videos
all_video0 = np.zeros(num_samples + size_frame, dtype="float32")
all_video1 = np.zeros(num_samples + size_frame, dtype="float32")
all_video2 = np.zeros(num_samples + size_frame, dtype="float32")
all_video3 = np.zeros(num_samples + size_frame, dtype="float32")
all_video4 = np.zeros(num_samples + size_frame, dtype="float32")
all_video5 = np.zeros(num_samples + size_frame, dtype="float32")

#print(all_video.shape)

# For each video.

for idx, path in enumerate(x_name):
    dr_counter = 0
    # Gather all its frames and add a batch dimension.
    roth=rootpath+path+'.mp4'
    
    cap = cv2.VideoCapture(roth)
    frames0 = []
    frames1 = []
    frames2 = []
    frames3 = []
    frames4 = []
    frames5 = []
    try:
        while (True):
            ret, frame = cap.read()
            if ret:
                #frame = crop_center_squar9/9e(frame)
                frame = cv2.resize(frame,  [height,width]) #size_frame[1:3])
                #frame = frame[:, :, [0, 0, 2]]
                if dr_counter%downsamplerate==0:
                    frames0.append(frame)
                if dr_counter%downsamplerate==1:
                    frames1.append(frame)
                if dr_counter%downsamplerate==2:
                    frames2.append(frame) 
                if dr_counter%downsamplerate==3:
                    frames3.append(frame)
                if dr_counter%downsamplerate==4:
                    frames4.append(frame) 
                if dr_counter%downsamplerate==5:
                    frames5.append(frame)   
               
                dr_counter+=1
                if len(frames5) == MAX_SEQ_LENGTH:
                    break
            else:
                break    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        frames0=np.array(frames0)
        frames1=np.array(frames1) 
        frames2=np.array(frames2)
        frames3=np.array(frames3)
        frames4=np.array(frames4)
        frames5=np.array(frames5)
       
    #print(frames.shape)
    #print(idx)
    all_video0[idx,...]=frames0
    all_video1[idx,...]=frames1
    all_video2[idx,...]=frames2 
    all_video3[idx,...]=frames3
    all_video4[idx,...]=frames4 
    all_video5[idx,...]=frames5
  
del frames0, frames1, frames2, frames3, frames4, frames5
    #all_frames = frames[None, ...]
    #print(all_frames.shape)    


# In[16]:


#train_data, train_labels = (frame_features, frame_masks),labels 
#test_data, test_labels = prepare_all_videos(test_df, "test")
# In[ ]:

filepath='/home/gadmin/Desktop/farshid/savic/bestmodel2.h5'
checkpoint=ModelCheckpoint(filepath=filepath,monitor='val_loss',verbose=1,
save_weights_only=True, save_best_only=True)
#model.fit(X , Y, epochs=100, batch_size=3, verbose=0)
estimator = KerasRegressor(model_pred_RNN, validation_split=0.15, validation_batch_size=1, epochs=500, batch_size=1, random_state=5,  shuffle=False, callbacks=[checkpoint])
#loo = LeaveOneOut()
#kf = KFold(n_splits=5)

# method 1
#history = cross_val_score(estimator,all_video ,labels, cv=kf,scoring='neg_mean_squared_error' )
#print("Baseline: %.2f (%.2f) MSE" % (history.mean(), history.std()))

#read_file1 = pd.read_excel("/home/gadmin/Desktop/farshid/AB_region/100_TT_example.xlsx")

#train = read_file1.get(read_file1.columns[2])
#train_index  = enumerate(train)

#print(train_index)


#val = read_file1.get(read_file1.columns[3])
#val_index  = enumerate(val)

#print(val_index)

#train_index = [0,1,2,4,5,6,7,10,11,12,15,16,18,19,20,21,24,25,26,27,29,30,31,32,33,34,37,38,42,43,44,45,46,47,49,
#50,51,52,53,54,55,56,57,58,59,60,61,62,65,66,67,68,69,70,74,75,76,77,78,81,82,83,84,85,86,88,89,90,
#93,94,95,96,97]
#val_index = [3,8,9,13,14,17,22,23,28,35,36,39,41,48,63,64,71,72,73,79,80,87,91,92,98,99]


# method 2
cv_mse = []
cv_mape = []
cv_mae = []
#for train_index, val_index in kf.split(all_video):
#    print(train_index)

all_video=np.concatenate((all_video0,all_video1,all_video2,all_video3,all_video4,all_video5))
labels=np.concatenate((labels,labels,labels,labels,labels,labels))
print(np.shape(np.concatenate((all_video0,all_video1,all_video2,
all_video3,all_video4,all_video5))))
del all_video0, all_video1, all_video2, all_video3, all_video4, all_video5

train_index , test_index , train_labels , test_labels = train_test_split(all_video ,labels ,test_size=0.2 ,random_state=5)
del all_video, labels
#print(train_labels[1:15])
#print(test_labels[1:15])

history_fit = estimator.fit(train_index , train_labels)
del train_index, train_labels




#plt.plot(history.history_['loss'])
#print(history.history.keys())
# summarize history for loss
plt.plot(history_fit.history_['loss'],'.-k')
plt.plot(history_fit.history_['val_loss'],'m')
#plt.title('model loss')
plt.ylabel(r'$MAE\,(mm.h^{-1})$')
plt.xlabel(r'$Epoch$')
plt.legend([r'$Train$', '$validation$'], loc='upper right')
plt.savefig('/home/gadmin/Desktop/farshid/savic/MAE.png')
plt.show()

estimator.model.load_weights(filepath=filepath)
pred = estimator.predict(test_index)



# In[18]:


y_pred = np.array(pred)
#print(y_pred-bias_label)
y_test = np.array(test_labels)
#print(y_test-bias_label)
mse = np.square(np.subtract( y_test , y_pred )).mean()
print('mse=',mse)

 
mape = np.mean(np.abs((y_test - y_pred) / (y_test)))*100 
print('mape=',mape)

    
mae = np.mean(np.abs(y_test - y_pred))*1
print('mae=',mae)

 
R_square = r2_score(y_test , y_pred)
print('R_square=',R_square)
 
nse = he.evaluator(he.nse, y_pred, y_test)
print("NSE =",nse)

kge, r, alpha, beta = he.evaluator(he.kge, y_pred, y_test) 
print("KGE = ",kge)
 
plt.scatter(y_test , y_pred)
plt.show()  

y_test1=np.delete(y_test,np.where(y_test <= 6))
y_pred1=np.delete(y_pred,np.where(y_test <= 6))
    
mse1 = np.square(np.subtract( y_test1 , y_pred1 )).mean()
print('mse1=',mse1)
 
mape1 = np.mean(np.abs((y_test1 - y_pred1) / (y_test1)))*100 
print('mape1=',mape1)
    
mae1 = np.mean(np.abs(y_test1 - y_pred1))*1
print('mae1=',mae1)
 
R_square1 = r2_score(y_test1 , y_pred1)
print('R_square1=',R_square1)

nse1 = he.evaluator(he.nse, y_pred1, y_test1)
print("NSE =",nse)

kge1, r, alpha, beta = he.evaluator(he.kge, y_pred1, y_test1) 
print("KGE = ",kge1)

plt.scatter(y_test1 , y_pred1 , marker='v' , color='m')
plt.ylabel(r'$Prediction\,(mm.h^{-1})$')
plt.xlabel(r'$Measured\,(mm.h^{-1})$')
plt.plot([0,70],[0,70], 'k')
plt.savefig('/home/gadmin/Desktop/farshid/savic/P-M.png')
plt.show() 

#plt.title('mse')
#plt.savefig('/home/gadmin/Desktop/farshid/AB_region/mse.jpg')
#plt.show()

#plt.plot(cv_mape,'o-m')
#plt.title('mape')
#plt.savefig('/home/gadmin/Desktop/farshid/AB_region/mape.jpg')
#plt.show()





# In[ ]:




