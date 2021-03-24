import os
import shutil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy

data_origin_root = '/home/ubuntu/Desktop/style-transfer-antonio/data/dataset'
dataroot = '/home/ubuntu/Desktop/style-transfer-antonio/data/dataset/images/segmentation'
target_root = '/home/ubuntu/Desktop/style-transfer-antonio/data/junk_data'
products = os.listdir(dataroot)
plt.show()
for  i, product in enumerate(products):
    colors = os.listdir(os.path.join(dataroot, product))
    for j, color in enumerate(colors):
        items = os.listdir(os.path.join(dataroot, product, color))
        for k, item in enumerate(items):
            base = mpimg.imread(os.path.join(data_origin_root,'images','base', product, color, item))
            mask = mpimg.imread(os.path.join(data_origin_root,'images','mask', product, color, item))
            img = numpy.multiply(base * mask)
            plt.imshow(img)
            plt.show()
            print(f'product {product}  진행 현황 {i}/{len(products)} \nproduct {color}  진행 현황 {j}/{len(colors)} \n product {item}  진행 현황 {k}/{len(items)} ')
            opinion = input('좋은 데이터라면 그냥 enter키를, 지우려면 x 을 입력후 enter키를 입력해주세요')
            if opinion == 'y':
                shutil.move(os.path.join(dataroot, product, color, item), os.path.join(target_root, 'images', 'segmentation'))
                shutil.move(os.path.join(data_origin_root, 'clothes','base',product, color, item[:-3], 'jpg'), os.path.join(target_root, 'clothes','base',product, color, item[:-3], 'jpg'))
                shutil.move(os.path.join(data_origin_root, 'clothes', 'mask',product, color, item), os.path.join(target_root, 'clothes', 'mask',product, color, item))
                shutil.move(os.path.join(data_origin_root, 'images', 'base', product, color, item[:-3], 'jpg'), os.path.join(target_root, 'images', 'base', product, color, item[:-3], 'jpg'))
                shutil.move(os.path.join(data_origin_root, 'images', 'mask', product, color, item), os.path.join(target_root, 'images', 'mask', product, color, item))
            else:
                print('제대로 입력해줘잉')

            plt.close(img)