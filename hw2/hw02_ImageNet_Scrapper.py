import torchvision
import torch.utils.data
import glob, os, numpy, PIL, argparse, requests, logging, json

from PIL import Image
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL

parser = argparse.ArgumentParser(description='HW02 Task1')
parser.add_argument('--subclass_list', nargs='*', type=str, required=True)
parser.add_argument('--images_per_subclass', type=int, required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--main_class', type=str, required=True)
parser.add_argument('--imagenet_info_json', type=str, required=True)
args, args_other = parser.parse_known_args()


def get_image(img_url, class_folder):
    if len(img_url) <= 1:
        # url is useless, do something
        return 0

    try:
        img_resp = requests.get(img_url, timeout=1)
    except ConnectionError:
        # handle
        logging.debug(f"Connection Error for url {img_url}")
        return 0
    except ReadTimeout:
        # handle
        logging.debug(f"Read Timeout for url {img_url}")
        return 0
    except TooManyRedirects:
        # handle
        logging.debug(f"Too many redirects {img_url}")
        return 0
    except MissingSchema:
        # handle
        return 0
    except InvalidURL:
        # handle
        return 0
    
    if not 'content-type' in img_resp.headers:
        # missing content, do something
        return 0
    if not 'image' in img_resp.headers['content-type']:
        # the url doesn't have any image, do something
        logging.debug("Not an image")
        logging.debug(img_resp.headers['content-type'])
        return 0
    if (len(img_resp.content) < 1000):
        # ignore images < 1kb
        logging.debug(f"Image size {len(img_resp.content)}")
        return 0
    
    img_name = img_url.split('/')[-1]
    img_name = img_name.split("?")[0]

    if (len(img_name) <= 1):
        # missing image name
        logging.debug(f"Image name missing")
        return 0
    if not 'flickr' in img_url:
        # missing non-flickr images are difficult to handle, do something
        logging.debug(f"Non-flickr image")
        return 0
    
    img_file_path = os.path.join(class_folder, img_name)
    if os.path.isfile(img_file_path):
        return 0
    logging.debug(f"Saving image in {img_file_path}")

    with open(img_file_path, 'wb') as img_f:
        img_f.write(img_resp.content)
    
    # resize image to 64*64
    im = Image.open(img_file_path)

    if im.mode != 'RGB':
        im = im.convert(mode='RGB')
    
    im_resized = im.resize((64, 64), Image.BOX)
    # overwrite original image with downsampled image
    im_resized.save(img_file_path)
    return 1


def main():
    # the_url contains the required url to obtain the full list using an identifier
    prefix = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='
    file = open('./imagenet_class_info.json')
    data = json.load(file)

    class_folder_path = os.path.join(args.data_root, args.main_class)
    if not os.path.isdir(class_folder_path):
        os.makedirs(class_folder_path)

    wnids = []
    for k, v in data.items():
        if v['class_name'] in args.subclass_list:
            wnids.append(k)
    print(wnids)
    for wnid in wnids:
        the_list_url = prefix + wnid
        resp = requests.get(the_list_url)
        urls = [url.decode('utf-8') for url in resp.content.splitlines()]
        num_images = 0

        for url in urls:
            if num_images >= args.images_per_subclass:
                print(num_images)
                break
            num_images += get_image(url, class_folder_path)

main()