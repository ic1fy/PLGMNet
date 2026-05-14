import numpy as np
import random
import os


def create_txt(save_path):
    with open(save_path, 'w') as f: 
        imgs_name = os.listdir(images_path)
        print(len(imgs_name))

        for img_name in imgs_name:
            mask_name = img_name.replace('.jpg', '.png')
            f.write(f'{images_path}{img_name} {ground_truth_path}{mask_name} \n')
        
if __name__ == "__main__":
    ground_truth_path = 'data/Test/Mask/'
    images_path = 'data/Test/Image/'
    create_txt('data/test.txt')