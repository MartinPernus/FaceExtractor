
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from FaceExtractor import Extractor


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

img = Image.open('img.jpg')
img = to_tensor(img).to(device)

extractor = Extractor().to(device)
img_new = extractor(img)
save_image(img_new.cpu(), 'img_extracted.jpg')

