import numpy as np
import os
import cv2

def resize_grayscale_labels_train(directory, d1, d2):
    dataset = []
    count = 0
    for current_dir in os.walk(directory):
        for current_file in current_dir[2]:
            current_file_path = current_dir[0] + "/" + current_file

            label_index = int(current_dir[0][-1])
            label = np.zeros(10)
            label[label_index] = 1

            img = cv2.imread(current_file_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (d1, d2))

            example = []
            example.append(img)
            example.append(label)
            dataset.append(example)

            count += 1
            print('files processed: ', count)

    return np.array(dataset)


train_directory = 'G:/DL/distracted-driver-new/driver_imgs_list.csv/train'
new_train_directory = 'G:/DL/distracted-driver-new/driver_imgs_list.csv/new_train'

dataset_train = resize_grayscale_labels_train(train_directory, 224, 224)
print(dataset_train.shape)
print(dataset_train[0][0].shape)
print(dataset_train[0][1].shape)
np.save(new_train_directory + '/' + 'dataset_train', dataset_train)
