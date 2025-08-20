from math import ceil
import os
import logging
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass
import random
import lmdb
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from timm.data import create_transform

from src.clip import _tokenizer
from src.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


import json

def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            data.append({
                'item_id_1': record['item_id_1'],
                'item_title_1': record['item_title_1'],
                'item_image_1': record['item_image_1'],
                'item_id_2': record['item_id_2'],
                'item_title_2': record['item_title_2'],
                'item_image_2': record['item_image_2'],
                'tag': record['tag']
            })
    
    # Shuffle the data
    random.shuffle(data)
    
    return data

class JSONLDataset(Dataset):
    def __init__(self, jsonl_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        super(JSONLDataset, self).__init__()
        self.data = load_jsonl_data(jsonl_path)
        self.dataset_len = len(self.data)
        self.global_batch_size = 1
        self.split = split
        self.max_txt_length = max_txt_length
        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)
        self.resolution = resolution
        
        # Add placeholder image for broken images
        self.placeholder_image = self._create_placeholder_image(resolution)
        
        # Keep track of problematic images
        self.problematic_images = set()
        
    def _create_placeholder_image(self, resolution):
        """Create a simple placeholder image for broken images"""
        import numpy as np
        from PIL import Image
        
        # Create a gray placeholder image
        placeholder = np.ones((resolution, resolution, 3), dtype=np.uint8) * 128
        placeholder_image = Image.fromarray(placeholder.astype('uint8'))
        
        # Apply the transformation to convert it to tensor format
        return self.transform(placeholder_image)
        
    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            transform = create_transform(
                input_size=resolution,
                scale=(0.9, 1.0),
                is_training=True,
                color_jitter=None,
                auto_augment='original',
                interpolation='bicubic',
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:])
        else:
            transform = Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform

    def __len__(self):
        return self.dataset_len
        
    def _safe_open_image(self, image_path):
        """Safely open an image with proper error handling"""
        try:
            # Check if file exists and has content
            import os
            if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                if image_path not in self.problematic_images:
                    logging.warning(f"Image file is missing or empty: {image_path}")
                    self.problematic_images.add(image_path)
                return None
                
            # Try to open and process the image
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
            
        except (PIL.UnidentifiedImageError, OSError, IOError) as e:
            # Log the error (only once per image to avoid log flooding)
            if image_path not in self.problematic_images:
                logging.warning(f"Failed to load image {image_path}: {str(e)}")
                self.problematic_images.add(image_path)
            return None
        except Exception as e:
            # Catch any other unexpected errors
            if image_path not in self.problematic_images:
                logging.warning(f"Unexpected error loading {image_path}: {str(e)}")
                self.problematic_images.add(image_path)
            return None
    
    def __getitem__(self, index):
        record = self.data[index]
        
        # Process first item
        image1 = self._safe_open_image(record['item_image_1'])
        if image1 is None:
            image1 = self.placeholder_image
            
        text1 = tokenize([_preprocess_text(record['item_title_1'])], context_length=self.max_txt_length)[0]
    
        # Process second item
        image2 = self._safe_open_image(record['item_image_2'])
        if image2 is None:
            image2 = self.placeholder_image
            
        text2 = tokenize([_preprocess_text(record['item_title_2'])], context_length=self.max_txt_length)[0]
        
        # Get EOS indices
        eos_index1 = text1.numpy().tolist().index(_tokenizer.vocab['[SEP]']) if _tokenizer.vocab['[SEP]'] in text1.numpy().tolist() else 0
        eos_index2 = text2.numpy().tolist().index(_tokenizer.vocab['[SEP]']) if _tokenizer.vocab['[SEP]'] in text2.numpy().tolist() else 0
        eos_index1 = torch.tensor(eos_index1, dtype=torch.long)
        eos_index2 = torch.tensor(eos_index2, dtype=torch.long)
        
        # ID and tag
        item_id1 = torch.tensor(int(record['item_id_1']), dtype=torch.long)
        item_id2 = torch.tensor(int(record['item_id_2']), dtype=torch.long)
        tag = torch.tensor(int(record['tag']), dtype=torch.long)
    
        return image1, text1, eos_index1, image2, text2, eos_index2, tag, item_id1, item_id2     
    

def pad_dataset(dataset, global_batch_size):
    # 计算需要填充后的数据集长度
    original_len = dataset.dataset_len
    padded_len = ceil(original_len / global_batch_size) * global_batch_size
    
    # 修改数据集的长度属性
    dataset.dataset_len = padded_len
    dataset.global_batch_size = global_batch_size
    
    # 如果需要填充，直接在data中添加最后一个样本的副本
    if original_len < padded_len:
        # 重复最后一个元素进行填充
        for _ in range(padded_len - original_len):
            dataset.data.append(dataset.data[-1])

def drop_dataset(dataset, global_batch_size):
    # edit dataset.__len__() of the dataset
    dataset.dataset_len = floor(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: JSONLDataset
    epoch_id: int


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    dataset = JSONLDataset(
        db_path, 
        split="train" if is_train else "val",
        max_txt_length=max_txt_length,
        use_augment=args.use_augment if is_train else False,
        resolution=fetch_resolution(args.vision_model)
    )

    # pad the dataset splits using the beginning samples in the JSONL file
    # to make the number of samples enough for a full final global batch
    batch_size = args.batch_size if is_train else args.valid_batch_size
    global_batch_size = batch_size * torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    sampler.set_epoch(epoch_id if is_train else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers if is_train else args.valid_num_workers,
        sampler=sampler,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)



def get_data(args, epoch_id=0, max_txt_length=64):
    data = {}

    if args.train_data:
        data["train"] = get_dataset(
            args, 
            is_train=True,  
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id)

    if args.val_data:
        data["val"] = get_dataset(
            args, 
            is_train=False, 
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id)

    return data



if __name__ == "__main__":
    dataset = JSONLDataset(jsonl_path='/etc/ssd1/wangzihan11/same_item_data/part5_tmp_data.jsonl', split='train', max_txt_length=64, use_augment=True, resolution=224)
    print(len(dataset))
    print(dataset[0])
    image_1, text_1, e1,image_2, text_2 ,e2= dataset[0]
    print(image_1.shape, text_1.shape, image_2.shape, text_2.shape)