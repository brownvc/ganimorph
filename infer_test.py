from pathlib import Path

import argparse
import numpy as np
import skimage
import skimage.io
from tensorpack import SaverRestore, PredictConfig, OfflinePredictor

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help='path to trained model', required=True)
parser.add_argument('--input_image_path', help='path to load input image from', required=True, type=Path)
parser.add_argument('--output_image_path', help='path to save output images', required=True, type=Path)
args = parser.parse_args()

def resize_shortest(x, size):
    h, w = x.shape[:2]
    scale = size * 1.0 / min(h, w)
    if h < w:
        newh, neww = size, int(scale * w + 0.5)
    else:
        newh, neww = int(scale * h + 0.5), size
    return skimage.transform.resize(x, (newh, neww))
def crop_center(img,cropx,cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]
if __name__ == '__main__':
    pred_config = PredictConfig(
        model=Model(),
        session_init=SaverRestore(args.model_path),
        input_names=['inputB'],
        output_names=['gen/A/deconv3/output:0'],
    )
    predictor = OfflinePredictor(pred_config)

    image = skimage.io.imread(args.input_image_path)
    image = resize_shortest(image, int(128 * 1.18))
    image = crop_center(image,128,128)
    #image = crop_center(image, 128, 128)
    image = image.astype(np.float32)

    inputB = image.copy()[np.newaxis, ...]

    outputA, = predictor(inputB)
    outputA = (outputA[0].transpose((1, 2, 0)) * 255).astype(np.uint8)

    #args.output_image_path.mkdir(exist_ok=True, parents=True)
    skimage.io.imsave(str(args.output_image_path), outputA)
