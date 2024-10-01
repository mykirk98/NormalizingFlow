import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        base_transforms = [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

        # if is_train:
        #     self.image_transform = transforms.Compose(base_transforms)
        # else:
        #     self.image_transform = transforms.Compose(base_transforms)
        self.image_transform = transforms.Compose(base_transforms)

            # perturbations = [
            #     transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            # ]
            # self.image_transform = transforms.Compose(perturbations + base_transforms)

            # self.target_transform = transforms.Compose([
            #     transforms.Resize(input_size),
            #     transforms.ToTensor(),
            # ])

        if is_train:
            self.image_files = glob(pathname=os.path.join(root, category, "train", "good", "*.png"))
        else:
            self.image_files = glob(pathname=os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                )
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)

class MoldDataset(torch.utils.data.Dataset):
    def __init__(self, root, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                )
                target = self.target_transform(target)
                try: target = target[0:1,:,:]
                except: pass
            return image, target

    def __len__(self):
        return len(self.image_files)