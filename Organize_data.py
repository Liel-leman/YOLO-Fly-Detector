import random
import re
import glob
import pandas as pd
import os
import shutil
from utils import yolo_representation, splitter, tiler
from PIL import Image
import argparse

if __name__ == "__main__":

    # setup dir names
    crsPath = './hen_images/original'  # dir where images and annotations stored
    DestPath = './hen_images/yolo_representation'  # destination of where to store yolo representation files
    SlicedPath = './datasliced/'  # estination of where to store the Sliced yolo representation files (it will be already splited to train set and val set
    Image.MAX_IMAGE_PIXELS = 250000000  # our pictures are big thats why we need to extand the max size of image in PIL

    df = pd.read_csv(crsPath + "/boxes.csv")  # loading boxes file in csv format
    if not os.path.exists(DestPath):
        os.makedirs(DestPath)

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-size", type=int, default=2048, help="Size of a tile. Dafault: 2048")
    parser.add_argument("-ratio", type=float, default=0.8, help="Train/test split ratio. Dafault: 0.8")

    args = parser.parse_args()

    # storing files to corresponding yolo representation
    for (dirname, dirs, files) in os.walk(crsPath):
        for filename in files:
            if filename.endswith('.jpg'):
                shutil.copy(os.path.join(crsPath, filename), os.path.join(DestPath, filename))

                # get the objects for the same image
                df_exact_file = df[df["parent_image_file_name"] == filename]
                boxes = df_exact_file[['x', 'y', 'w', 'h']].to_numpy()

                fileLabelString = filename[:-4] + '.txt'  # get name of corresponding annotation file
                FileLabelContext = yolo_representation(os.path.join(crsPath, filename),
                                                       boxes)  # change from coco representation to yolo
                with open(os.path.join(DestPath, fileLabelString), "w") as f:
                    f.write(FileLabelContext)

                print(
                    f'{os.path.join(crsPath, filename)} made to be represented in yolo format at ->{os.path.join(DestPath, filename)}')

    #Generate tex file with class names:
    with open("/".join(crsPath.split('/')[:-1])+'/classes.names', 'w') as f:
        f.write("['fly']")



    imnames = glob.glob(f'{DestPath}//*.jpg')
    labnames = glob.glob(f'{DestPath}//*.txt')

    if len(imnames) == 0:
        raise Exception("Source folder should contain some images")
    elif len(imnames) != len(labnames):
        raise Exception("Dataset should contain equal number of images and txt files with labels")

    if not os.path.exists(SlicedPath):
        os.makedirs(SlicedPath)
    elif len(os.listdir(SlicedPath)) > 0:
        raise Exception("Target folder should be empty")

    # classes.names should be located one level higher than images
    # this file is not changing, so we will just copy it to a target folder
    upfolder = os.path.join(DestPath, '/..')
    target_upfolder = os.path.join(DestPath, '/..')
    if not os.path.exists(os.path.join(upfolder, 'classes.names')):
        print('classes.names not found. It should be located one level higher than images')
    else:
        shutil.copyfile(os.path.join(upfolder, 'classes.names'), os.path.join(target_upfolder, 'classes.names'))

    tiler(imnames, SlicedPath, args.size, ".jpg")
    splitter(SlicedPath, target_upfolder, ".jpg", args.ratio)
