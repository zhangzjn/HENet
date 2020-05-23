from torchvision.transforms import *
from PIL import Image, ImageDraw
import random
import math
import numpy as np
from torchvision import transforms


class RandomPolygonErasing(object):
    """ Randomly selects a polygon region in an image and erases its pixels.
            by zhangzjn, 2019.1.1
            See https://arxiv.org/pdf/2005.08812.pdf
        Args:
             probability: The probability that the Random Erasing operation will be performed.
             pt_num: The number of vertices that make up the random polygon.
             sl: Minimum proportion of erased area against input image.
             sh: Maximum proportion of erased area against input image.
             r1: Minimum aspect ratio of erased area.
             mean: Erasing value.
        """
    def __init__(self, probability=0.5, pt_num=20, sl=0.02, sh=0.45, r=0.35, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        assert pt_num>=3, 'pt_num less than 3 ...'
        self.pt_num = pt_num
        self.sl = sl
        self.sh = sh
        self.r = r
       
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        def generate_pt_list():
            while True:
                area = img.size()[1] * img.size()[2]
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r, 1 / self.r)
                r_w = int(round(math.sqrt(target_area / aspect_ratio)))
                r_h = int(round(math.sqrt(target_area * aspect_ratio)))
                pt_list = []
                if r_w >= img.size()[2] or r_h >= img.size()[1]:
                    continue
                center_pt = (random.randint(r_w//2, img.size()[2]-r_w//2), random.randint(r_h//2, img.size()[1]-r_h//2))
                pt_list.append(center_pt)

                w_min = max([center_pt[0] - r_w//2, 0])
                w_max = min([center_pt[0] + r_w//2, img.size()[2]])
                h_min = max([center_pt[1] - r_h//2, 0])
                h_max = min([center_pt[1] + r_h//2, img.size()[1]])
                for i in range(self.pt_num-1):
                    x = int(random.randint(w_min, w_max))
                    y = int(random.randint(h_min, h_max))
                    pt_list.append((x, y))
                return pt_list

        mask = Image.fromarray(np.zeros((img.shape[1], img.shape[2])))
        draw = ImageDraw.Draw(mask)
        pts = generate_pt_list()
        # for simplicity
        for i in range(self.pt_num-2):
            for j in range(0, self.pt_num):
                draw.polygon([pts[i], pts[i+1], pts[j]], fill=(1))
        mask = transforms.ToTensor()(mask)
        mask_neg = 1-mask
        for cnt in range(3):
            img[cnt] = img[cnt] * mask_neg + mask * self.mean[cnt]
        return img
