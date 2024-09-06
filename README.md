# Rain-Intensity-Estimation-from-RGB-videos
The main code is executed under the name: RNN_CNN_SV.py

In the main code the downsample rate is specified under the variable name "downsamplerate".

The path for EfficientNet model is specified by the variable "wp", which has to be changed. It can be downloaded from https://www.kaggle.com/datasets/mujawar/efficientnetb0-notop-h5

The "convert" folder contains 1-min rain videos, whose path is specified by the variable "rootpath". The convert folder can be downloaded from https://drive.google.com/file/d/1VlEinoqA_2NpXiTLEUrectGi20bZb49L/view?usp=sharing

The path to the excel file including the names of 1-min events is specified at line 115.

If you find the code interseting or have a question please contact me at: nfaraji@eng.ikiu.ac.ir


The code is related to this article:

Rajabi,F., Faraji, N. & Hashemi, M. An efficient video-based rainfall intensity estimation employing different recurrent neural network models. Earth Sci Inform 17, 2367–2380 (2024). https://doi.org/10.1007/s12145-024-01290-x
