"""Load CameraAgent image tensors in DataLoader workers."""

from PIL import Image
from torch.utils.data import Dataset

from .model import image_tensor


def load_image_tensor(path, image_size):
    """Match CameraModel's pixel limit for trusted local training images."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        return image_tensor(image, image_size)


class CameraImageDataset(Dataset):
    def __init__(self, records, image_size):
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        pixels = load_image_tensor(item["image"], self.image_size)
        return {"record": item["record"], "pixels": pixels}
