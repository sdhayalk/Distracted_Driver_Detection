import numpy as np
import os
import cv2

def resize_grayscale(directory, d1, d2):
    count = 0
    for current_dir in os.walk(directory):
        for current_file in current_dir[2]:
            current_file_path = current_dir[0] + "/" + current_file

            img = cv2.imread(current_file_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (d1, d2))
            cv2.imwrite(current_file_path, img)

            count += 1
            print('files processed: ', count)

def get_dataset(directory):
    dataset = []
    count = 0
    for current_dir in os.walk(directory):
        for current_file in current_dir[2]:
            current_file_path = current_dir[0] + "/" + current_file

            label_index = int(current_dir[0][-1])
            label = np.zeros(10, dtype=int)
            label[label_index] = 1

            img = cv2.imread(current_file_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            example = []
            example.append(img)
            example.append(label)
            dataset.append(example)

            count += 1
            print('files processed: ', count)
            # if count == 1000:
            #     return np.array(dataset)

    return np.array(dataset)


train_directory = 'G:/DL/distracted-driver-new/driver_imgs_list.csv/train'
test_directory = 'G:/DL/distracted-driver-new/driver_imgs_list.csv/test'

# resize_grayscale(train_directory, 224, 224)
# resize_grayscale(test_directory, 224, 224)
