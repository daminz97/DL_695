from pycocotools.coco import COCO
import torchvision
import torch.utils.data
import numpy as np
import glob, os, numpy, PIL, argparse, requests, logging, json
from PIL import Image
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL

parser = argparse.ArgumentParser(description='HW04 COCO downloader')
parser.add_argument('--root_path', required=True, type=str)
parser.add_argument('--coco_json_path', required=True, type=str)
parser.add_argument('--class_list', required=True, nargs='*', type=str)
parser.add_argument('--images_per_class', required=True, type=int)
args, args_other = parser.parse_known_args()

def get_image(img_url, class_folder):
    if len(img_url) <= 1:
        return 0
    try:
        img_resp = requests.get(img_url, timeout=1)
    except ConnectionError:
        return 0
    except ReadTimeout:
        return 0
    except TooManyRedirects:
        return 0
    except MissingSchema:
        return 0
    except InvalidURL:
        return 0
    if not 'content-type' in img_resp.headers:
        return 0
    if not 'image' in img_resp.headers['content-type']:
        return 0
    if (len(img_resp.content) < 1000):
        return 0
    img_name = img_url.split('/')[-1]
    img_name = img_name.split('?')[0]
    if (len(img_name) <= 1):
        return 0
    img_file_path = os.path.join(class_folder, img_name)
    if os.path.isfile(img_file_path):
        return 0
    with open(img_file_path, 'wb') as img_f:
        img_f.write(img_resp.content)
    im = Image.open(img_file_path)
    if im.mode != 'RGB':
        im = im.convert(mode='RGB')
    im_resized = im.resize((64,64), Image.BOX)
    im_resized.save(img_file_path)
    return 1

if __name__ == '__main__':
    img_folder_path = args.root_path
    coco = COCO(args.coco_json_path)
    catIds = coco.getCatIds(catNms=args.class_list)
    
    for class_name in args.class_list:
        class_folder_path = os.path.join(img_folder_path, class_name)
        if not os.path.isdir(class_folder_path):
            os.makedirs(class_folder_path)
        catIds = coco.getCatIds(catNms=class_name)
        imgIds = coco.getImgIds(catIds=catIds)
        num_images = 0
        while num_images < args.images_per_class:
            img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
            url = img['coco_url']
            num_images += get_image(url, class_folder_path)