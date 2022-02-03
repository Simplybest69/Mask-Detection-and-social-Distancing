# Mask-Detection-and-Social-Distancing 

This was done for the India Academia Connect AI Hackathon 2022 conducted by **NVIDIA**. 

The certificate for participation has been included in the repo, under the file name : IndiaAI_Hackathon_Sai Ganesh.pdf

The process or steps followed by the model are :

1. Detect the type of input, Image/Video stream.
2. Detect and find the location of the faces present in the stream.
3. Using the mask_det model, find the presence of a face mask on the found location.
4. Find the distance between people present in the stream.
5. Classify the risk category based on the above properties (Mask and distance).

The training of face mask detection has been done using various models such as resnet and vgg, the best 
performing(Accuracy) has been choosen for the hackathon.

### Face Mask Detection :

#### Dataset - https://drive.google.com/file/d/14AMs4ockT6Cp-Q5bevdXSYH_GDgLqb9x/view?usp=sharing

The dataset consists of 4092 images after data augmentation.

* With_Mask = 2162 Images
* Without_Mask = 1932 Images

The images are resized to *224 x 224 x 3* and the input pixel values are normalized between -1 and 1.

Since the data present was not enough to obtain a high enough accuracy, 
#### Additional data augmentation is done on the following parameters :

          * rotation_range
          * zoom_range
          * width_shift_range
          * shear_range
          * horizontal_flip
          * fill_mode
          
 - - - -
          
### Model Details:   

![image](https://user-images.githubusercontent.com/60337704/141739606-5a79c3a8-48e3-4030-a436-ec83cc2ed124.png)

     Input Shape : 224 x 224 x 3
     Number Of Layers : 159
     Trainable Parameters : 164,226

Size of Model after saving as a .hdf5 file : 12 MB

This small size makes it easier for the model to be deployable on most of the platforms with ease.

The final accuracy of the mask detection model is : 98%

![image](https://user-images.githubusercontent.com/60337704/141739905-b6ec19c9-1c72-41a5-a846-dbe8cf206283.png)

- - - -

### Social Distancing :

Finding distance b/w people wrt threshold.

![image](https://user-images.githubusercontent.com/60337704/141740267-18880031-24b1-4cd5-b354-886f0c940ca4.png)

- - - -

### Combination of the Above:

The face mask detection model and the estimation of distance between people have been combined to give the user a risk level based on the parameters.

* If a person is alone and not wearing a mask - **No Risk**. 
* If  more than one person : 
    * If both are apart and wearing a mask - **Low Risk**.
    * If both are near and wearing a mask - **Risk**.
    * If both are apart and not wearing a mask - **Risk**.
    * If both are near and not wearing a mask - **High Risk**.

![image](https://user-images.githubusercontent.com/60337704/141741264-3a04071c-09a3-4ab4-94e1-6e05fb80c63d.png)













  
  



