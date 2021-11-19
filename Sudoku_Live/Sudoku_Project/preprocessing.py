import torchvision.transforms as T
from PIL import Image

def convert_img_tensor(src_img):

    img = Image.open(src_img)

    convert_tensor = T.Compose([
    T.Resize(size=(28, 28)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])])

    return convert_tensor(img)