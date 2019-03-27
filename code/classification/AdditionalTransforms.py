import torchvision.transforms as transforms
import random
from PIL import ImageEnhance


class RandomRotation90:

    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img):
        return img.rotate(90) if random.random() < self.p else img


class RandomSaturation:

    def __init__(self, min_delta, max_delta):
        self.min_delta = min_delta
        self.max_delta = max_delta

    def __call__(self, img):
        scale = random.random() * (self.max_delta - self.min_delta) + self.min_delta
        return ImageEnhance.Color(img).enhance(scale)
