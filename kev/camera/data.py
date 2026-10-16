"""Load CameraAgent image tensors in DataLoader workers."""

from PIL import Image
from torch.utils.data import Dataset

from .model import image_tensor


class CameraImageDataset(Dataset):
    def __init__(self, records, image_size):
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        with Image.open(item["image"]) as image:
            pixels = image_tensor(image, self.image_size)
        return {"record": item["record"], "pixels": pixels}
