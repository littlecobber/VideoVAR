import csv
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from PIL import Image
from . import video_transforms
import chardet

import json
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import ipdb


def detect_encoding(file_path):
    with open(file_path,'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def get_transforms_video(resolution=256):
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform_video


def get_transforms_image(image_size=256):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform


class DatasetFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
            self,
            csv_path,
            num_frames=16,
            frame_interval=1,
            transform=None,
            root=None,
            max_samples=None,
    ):
        video_samples = []

        encoding = detect_encoding(csv_path)
        print(f"Detected encoding:{encoding}")

        with open(csv_path, "r", encoding=encoding) as f:
            reader = csv.reader(f)
            csv_list = list(reader)
            # print(f"CSV Head: {csv_list[:5]}")  # 打印前5行

        if max_samples is not None:
            csv_list = csv_list[1:max_samples + 1]
        else:
            csv_list = csv_list[1:]

        for vid in csv_list[1:]:  # no csv head
            vid_name = vid[0]
            vid_path = os.path.join(root, vid_name)

            vid_caption = vid[1]
            if os.path.exists(vid_path):
                video_samples.append([vid_path, vid_caption])

        self.samples = video_samples

        self.is_video = True
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.root = root

    def getitem(self, index):
        sample = self.samples[index]
        path = sample[0]
        text = sample[1]

        if self.is_video:
            is_exit = os.path.exists(path)
            if is_exit:
                vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                total_frames = len(vframes)
            else:
                total_frames = 0
            loop_index = index
            while (total_frames < self.num_frames or is_exit == False):
                loop_index += 1
                if loop_index >= len(self.samples):
                    loop_index = 0
                sample = self.samples[loop_index]
                path = sample[0]
                text = sample[1]

                is_exit = os.path.exists(path)
                if is_exit:
                    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec",
                                                                       output_format="TCHW")
                    total_frames = len(vframes)
                else:
                    total_frames = 0
            #  video exits and total_frames >= self.num_frames

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                    end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

            video = vframes[frame_indice]
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    data_path = r'/projectnb/ec720prj/DenseCap/vaex/OpenVid/OpenVid-1M-108part.csv'
    root = r'/projectnb/ec720prj/DenseCap/vaex/OpenVid/video'
    # data_path = r'D:\VideoVAR\OpenVid_part108\data\train\OpenVid-1M-108part.csv'
    # root = r'D:\VideoVAR\OpenVid_part108\video'
    dataset = DatasetFromCSV(
        data_path,
        transform=get_transforms_video(),
        num_frames=16,
        frame_interval=3,
        root=root,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=1
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    # for video_data in loader:
    #     print(video_data)

    # for i, video_data in enumerate(loader):
    #     if i >= 5:
    #         break
    #     print(f"Data {i + 1}:")
    #     for key, value in video_data.items():
    #         if isinstance(value, torch.Tensor):
    #             print(f"  {key}: {value.shape}")
    #         elif isinstance(value, list):
    #             print(f"  {key}: list of length {len(value)}")
    #         elif isinstance(value, str):
    #             print(f"  {key}: {value}")
    #         else:
    #             print(f"  {key}: {type(value)}")