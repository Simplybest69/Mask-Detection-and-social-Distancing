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


  
  



