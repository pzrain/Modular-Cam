import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from modularcam.utils.util import zero_rank_print



class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.control_image_transforms = transforms.Compose([
            transforms.RandomResizedCrop(
                (sample_size[0], sample_size[1]), (1.0, 1.0), 
                ratio=(sample_size[0]/sample_size[1], sample_size[1]/sample_size[0])
            )
            # transforms.ToTensor(),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]

        control_image = pixel_values[0].unsqueeze(0)
        return pixel_values, name, control_image

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, control_image = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.control_image_transforms(pixel_values)
        control_image = self.control_image_transforms(control_image)
        sample = dict(pixel_values=pixel_values, text=name, control_image=control_image)
        return sample



if __name__ == "__main__":
    from modularcam.utils.util import save_videos_grid

    dataset = WebVid10M(
        csv_path="webvid/results_20000_train_sample.csv",
        video_folder="webvid/videos_train",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=False,
    )
    # import pdb
    # pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, )
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]), batch["control_image"].shape)
        exit(1)
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)
