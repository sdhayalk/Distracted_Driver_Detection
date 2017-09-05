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
I implement 'GoogLeNet' architecture with Inception modules [2], which is a deep convolutional neural network that excels in image based classification. GoogLeNet, developed by researchers at Google, has been the winner of Imagenet ILSVRC 2014 challenge [2]. To enhance generalization, avoid over-fitting and a bit faster training, I added Dropout [3] and Batch Normalization [4] layers between layers.

### Dataset:
The dataset was obtained from Distracted Driver Dataset competition from Kaggle [1]. The dataset was well balanced across all classes, making things a bit easier. I first resize the dataset to 224x224, and grayscale it, thus getting only one channel. Also, normalize it between 0 and 1.

Example images obtained from Kaggle:
[Driver texting, calling, etc.](https://kaggle2.blob.core.windows.net/competitions/kaggle/5048/media/drivers_statefarm.png)

## Results:
Achieved an accuracy of 93 % using about 20000 training samples and 2400 testing samples. The code was run on FloydHub [5].

### References:
[1] [Distracted Driver Dataset competition from Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
[2] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
[3] Ioffe, S., & Szegedy, C. (2015, June). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning (pp. 448-456).
[4] Srivastava, N., Hinton, G. E., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. Journal of machine learning research, 15(1), 1929-1958.
[5] [FloydHub](https://www.floydhub.com/)
