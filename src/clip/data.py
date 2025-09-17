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
    data_by_leaf = {}
    all_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            # 保持原有字段
            data_record = {
                'item_id_1': record['item_id_1'],
                'item_title_1': record['item_title_1'],
                'item_image_1': record['item_image_1'],
                'item_id_2': record['item_id_2'],
                'item_title_2': record['item_title_2'],
                'item_image_2': record['item_image_2'],
                'tag': record['tag'],
                'spu_id_1': record['spu_id_1'],  # 需要这个字段做batch内去重
                'spu_id_2': record['spu_id_2'],  # 需要这个字段做batch内去重
                'leaf_id_1': record['leaf_id_1']  # 用于分组
            }
            
            leaf_id_1 = record['leaf_id_1']
            if leaf_id_1 not in data_by_leaf:
                data_by_leaf[leaf_id_1] = []
            data_by_leaf[leaf_id_1].append(data_record)
            all_data.append(data_record)
    
    # 保留原有的随机打乱
    random.shuffle(all_data)
    # 对每个leaf_id组内的数据也进行随机打乱
    for leaf_data in data_by_leaf.values():
        random.shuffle(leaf_data)
    
    return data_by_leaf, all_data

class JSONLDataset(Dataset):
    def __init__(self, jsonl_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        super(JSONLDataset, self).__init__()
        self.data_by_leaf, self.all_data = load_jsonl_data(jsonl_path)  # 修改这里
        self.dataset_len = len(self.all_data)
        self.global_batch_size = 1
        self.split = split
        self.max_txt_length = max_txt_length
        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)
        
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

class BatchSampler:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.data_by_leaf = dataset.data_by_leaf
        self.all_data = dataset.all_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.leaf_ids = list(self.data_by_leaf.keys())
        
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.leaf_ids)
        
        for leaf_id in self.leaf_ids:
            leaf_data = self.data_by_leaf[leaf_id]
            if len(leaf_data) == 0:
                continue
            
            # 对当前leaf_id的数据进行打乱
            if self.shuffle:
                leaf_data = leaf_data.copy()
                random.shuffle(leaf_data)
            
            # 组成batches
            i = 0
            while i < len(leaf_data):
                batch = []
                used_spu_ids_1 = set()  # 记录当前batch中已使用的第一个位置的spu_id
                
                # 先添加当前leaf_id的数据
                while i < len(leaf_data) and len(batch) < self.batch_size:
                    data = leaf_data[i]
                    # 检查这条数据的spu_id_1是否与已有batch中的spu_id_2有冲突
                    conflict = False
                    for b in batch:
                        if data['spu_id_1'] == b['spu_id_2']:
                            conflict = True
                            break
                    if not conflict:
                        batch.append(data)
                    i += 1
                
                # 如果batch还未满，从其他数据中随机补充
                while len(batch) < self.batch_size:
                    # 随机选择一条数据
                    data = random.choice(self.all_data)
                    # 检查是否有冲突
                    conflict = False
                    for b in batch:
                        if data['spu_id_1'] == b['spu_id_2']:
                            conflict = True
                            break
                    if not conflict:
                        batch.append(data)
                
                # 转换为索引
                indices = [self.all_data.index(x) for x in batch]
                yield indices
    
    def __len__(self):
        num_batches = 0
        for leaf_data in self.data_by_leaf.values():
            num_batches += len(leaf_data) // self.batch_size
        return max(1, num_batches)  # 至少返回1,避免除0错误
##新加的
        
    def __getitem__(self, index):
        record = self.data[index]
        
        # 处理第一组数据
        image1 = Image.open(record['item_image_1']).convert('RGB')
        image1 = self.transform(image1)
        text1 = tokenize([_preprocess_text(record['item_title_1'])], context_length=self.max_txt_length)[0]
    
        # 处理第二组数据
        image2 = Image.open(record['item_image_2']).convert('RGB')
        image2 = self.transform(image2)
        text2 = tokenize([_preprocess_text(record['item_title_2'])], context_length=self.max_txt_length)[0]
        
        # 索引相关的值
        eos_index1 = text1.numpy().tolist().index(_tokenizer.vocab['[SEP]']) if '[SEP]' in text1.numpy().tolist() else 0
        eos_index2 = text2.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        eos_index1 = torch.tensor(eos_index1, dtype=torch.long)
        eos_index2 = torch.tensor(eos_index2, dtype=torch.long)
        
        # ID和tag
        item_id1 = torch.tensor(int(record['item_id_1']), dtype=torch.long)
        item_id2 = torch.tensor(int(record['item_id_2']), dtype=torch.long)
        tag = torch.tensor(int(record['tag']), dtype=torch.long)
    
        return image1, text1, eos_index1, image2, text2, eos_index2, tag, item_id1, item_id2
    

def pad_dataset(dataset, global_batch_size):
    original_len = dataset.dataset_len
    padded_len = ceil(original_len / global_batch_size) * global_batch_size
    
    dataset.dataset_len = padded_len
    dataset.global_batch_size = global_batch_size
    
    if original_len < padded_len:
        last_record = dataset.all_data[-1]
        for _ in range(padded_len - original_len):
            dataset.all_data.append(last_record)

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

    batch_size = args.batch_size if is_train else args.valid_batch_size
    batch_sampler = BatchSampler(dataset, batch_size=batch_size, shuffle=True)
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers if is_train else args.valid_num_workers,
        pin_memory=True
    )

    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(batch_sampler)

    return DataInfo(dataloader, None, dataset, epoch_id)



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
    dataset = JSONLDataset(jsonl_path='', split='train', max_txt_length=64, use_augment=True, resolution=224)
    print(len(dataset))
    print(dataset[0])
    image_1, text_1, e1,image_2, text_2 ,e2= dataset[0]
    print(image_1.shape, text_1.shape, image_2.shape, text_2.shape)