# Mask-and-social-Distancing detection

This was done for the India Academia Connect AI Hackathon conducted by **NVIDIA**. 

### Dataset - 

#### Face Mask Detection :
The dataset consists of 4092 images.

* With_Mask : 2162 images
* Without_Mask : 1932 images

The images are resized to *224 x 224 x 3* and the input pixel values are scaled between -1 and 1.

The data is split into 80% for training and remaining 20% for testing.

#### Additional data augmentation is done on the following parameters :

          * rotation_range
          * zoom_range
          * width_shift_range
          * shear_range
          * horizontal_flip
          * fill_mode
          
### Model Details:   

![image](https://user-images.githubusercontent.com/60337704/141739606-5a79c3a8-48e3-4030-a436-ec83cc2ed124.png)

     Input Shape : 224 x 224 x 3
     Number Of Layers : 159
     Trainable Parameters : 164,226

Size of Model after saving as a .hdf5 file : 12 MB

This small size makes it easier for the model to be deployable on most of the platforms with ease.

The final accuracy of the mask detection model is : 98%

![image](https://user-images.githubusercontent.com/60337704/141739905-b6ec19c9-1c72-41a5-a846-dbe8cf206283.png)





  
  



