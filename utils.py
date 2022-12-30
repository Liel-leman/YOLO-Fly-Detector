
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import glob

import os
import random

def yolo_representation(imgPath,boxs, labels=None):
    s = ''
    if not labels:
        labels = [0 for i in range(boxs.shape[0])]
    # coco_rep = np.column_stack((labels, boxs))

    img = Image.open(imgPath)
    width = img.width
    height = img.height

    yolo_rep = [convert_bbox_coco2yolo(width, height, box) for box in boxs]

    for i,box in enumerate(yolo_rep):
        x,y,w,h = box
        label = labels[i]
        s += f'{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n'



    return s[:-1]


def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format:
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format:
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """

    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]


def tiler(imnames, newpath, slice_size, ext):
    for imname in imnames:
        im = Image.open(imname)
        imr = np.array(im, dtype=np.uint8)
        height = imr.shape[0]
        width = imr.shape[1]
        labname = imname.replace(ext, '.txt')
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

        # we need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height

        boxes = []

        # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w'] / 2
            y1 = (height - row[1]['y1']) - row[1]['h'] / 2
            x2 = row[1]['x1'] + row[1]['w'] / 2
            y2 = (height - row[1]['y1']) + row[1]['h'] / 2

            boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))

        counter = 0
        print('Image:', imname)
        # create tiles and find intersection with bounding boxes for each tile
        for i in range((height // slice_size)):
            for j in range((width // slice_size)):
                x1 = j * slice_size
                y1 = height - (i * slice_size)
                x2 = ((j + 1) * slice_size) - 1
                y2 = (height - (i + 1) * slice_size) + 1

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                for box in boxes:
                    if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])

                        if not imsaved:
                            sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
                            sliced_im = Image.fromarray(sliced)
                            filename = imname.split('\\')[-1]
                            slice_path = newpath + "/" + filename.replace(ext, f'_{i}_{j}{ext}')
                            slice_labels_path = newpath + "/" + filename.replace(ext, f'_{i}_{j}.txt')
                            print(slice_path)
                            sliced_im.save(slice_path)
                            imsaved = True

                            # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope

                        # get central point for the new bounding box
                        centre = new_box.centroid

                        # get coordinates of polygon vertices
                        x, y = new_box.exterior.coords.xy

                        # get bounding box width and height normalized to slice size
                        new_width = (max(x) - min(x)) / slice_size
                        new_height = (max(y) - min(y)) / slice_size

                        # we have to normalize central x and invert y for yolo format
                        new_x = (centre.coords.xy[0][0] - x1) / slice_size
                        new_y = (y1 - centre.coords.xy[1][0]) / slice_size

                        counter += 1

                        slice_labels.append([box[0], new_x, new_y, new_width, new_height])

                if len(slice_labels) > 0:
                    slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                    print(slice_df)
                    slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')



def splitter(target, target_upfolder, ext, ratio):
    imnames = glob.glob(f'{target}/*{ext}')
    names = [name.split('\\')[-1] for name in imnames]

    # split dataset for train and test

    train = []
    val = []

    for dataset in ['val', 'train']:
        for type_folder in ['images', 'labels']:
            adder = f'{dataset}/{type_folder}'
            directory = target+adder
            if not os.path.exists(directory):
                os.makedirs(directory)

    for name in names:
        locFile = os.path.join(target, name)
        locLabel = locFile.replace('jpg', 'txt')
        if random.random() > ratio:
            shutil.move(locFile, target+'val/images/'+locLabel.split('/')[-1])
            shutil.move(locLabel, target+'val/labels/'+locLabel.split('/')[-1])
            val.append(os.path.join(target+'val/labels', locLabel.split('/')[-1]))
        else:
            shutil.move(locFile, target+'train/images/'+locLabel.split('/')[-1])
            shutil.move(locLabel, target+'train/labels/'+locLabel.split('/')[-1])
            train.append(os.path.join(target+'train/labels', locLabel.split('/')[-1]))
    print('train:', len(train))
    print('val:', len(val))

    # we will put val.txt, train.txt in a folder one level higher than images

    # save train part
    with open(f'{target}/train.txt', 'w') as f:
        for item in train:
            f.write("%s\n" % item)

    # save test part
    with open(f'{target}/val.txt', 'w') as f:
        for item in val:
            f.write("%s\n" % item)