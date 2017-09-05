# Distracted Driver Detection In Cars

## Overview:
Implemented deep convolutional neural network 'GoogLeNet' architecture with Inception modules that can detect the driver's activity or distraction such as texting, drinking, talking, etc., which can be used to aid autonomous driving agent about driver's state.

## Detail:
This project and dataset is derived from Distracted Driver Dataset competition from Kaggle [1]. According to the CDC motor vehicle safety division, one in five car accidents is caused by a distracted driver. Sadly, this translates to 425,000 people injured and 3,000 people killed by distracted driving every year [1]. The aim of this project is to identify what other activity the driver is doing while driving (whether he is distracted). The set of possible activites are (10 activities/classes): 
```bashp
1. c0: normal driving
2. c1: texting - right
3. c2: talking on the phone - right
4. c3: texting - left
5. c4: talking on the phone - left
6. c5: operating the radio
7. c6: drinking
8. c7: reaching behind
9. c8: hair and makeup
10. c9: talking to passenge
```
Identifying whether the user is distracted can be very useful information for the autonomous agent in self-driving cars, and aid or alert the driver if necessary. This helps keeping the driver as well as the passengers safe from mishap due to driver's lack of attention.

## Solution:
I implement 'GoogLeNet' architecture with Inception modules [2], which is a deep convolutional neural network that excels in image based classification. GoogLeNet, developed by researchers at Google, has been the winner of Imagenet ILSVRC 2014 challenge [2]. To enhance generalization and a bit faster training, I added Dropout and Batch Normalization layers between layers. Batch Normalization also solved the issue of predicting only one class for all examples.

### Dataset:
The dataset was obtained from Distracted Driver Dataset competition from Kaggle [1]. The dataset was well balanced across all classes, making things a bit easier. I first resize the dataset to 224x224, and grayscale it, thus getting only one channel. Also, normalize it between 0 and 1.

Example images obtained from Kaggle:
[Driver texting, calling, etc.](https://kaggle2.blob.core.windows.net/competitions/kaggle/5048/media/drivers_statefarm.png)
