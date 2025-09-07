import json
import numpy as np
import os.path as osp
import os
import torch
import random
from torch.utils.data import Dataset
from utils_prompt import *
from tqdm import trange

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
}


def load_data_std(args):
    problems = json.load(open(osp.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(open(osp.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits[args.train_split]
    val_qids = pid_splits[args.val_split]
    test_qids = pid_splits[args.test_split]

    print(f"Number of train problems: {len(train_qids)}")
    print(f"Number of val problems: {len(val_qids)}")
    print(f"Number of test problems: {len(test_qids)}")

    qids = {'train': train_qids, 'val': val_qids, 'test': test_qids}
    return problems, qids


def load_data_img(args):
    problems = json.load(open(osp.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(open(osp.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]
    name_maps = json.load(open(osp.join(args.data_root, 'vision_features/name_map.json')))
    if args.r_image == "Yes":
        r_name_maps = json.load(open(osp.join(args.data_root, 'vision_features/r_name_map.json')))

    if args.img_type == "resnet":
        image_features = np.load(osp.join(args.data_root, 'vision_features/resnet.npy'))
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load(osp.join(args.data_root, 'vision_features/clip.npy'))
    elif args.img_type == "detr":
        image_features = np.load(osp.join(args.data_root, 'vision_features/detr.npy'))
        if args.r_image == "Yes":
            r_image_features = np.load(osp.join(args.data_root, 'vision_features/r_detr.npy'))
    else:
        image_features = np.load(osp.join(args.data_root, 'vision_features/detr.npy'))

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits[args.train_split]
    train_rqids = None
    if args.prompt_format in ['QCM-LE', 'QCM-E']:
        file_path = "data/train_rqids.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                train_rqids = json.load(file)
        else:
            train_rqids = [[]] * len(train_qids)
            for i in trange(len(train_qids)):
                for j in range(i + 1, len(train_qids)):
                    instance_i = problems[train_qids[i]]
                    instance_j = problems[train_qids[j]]
                    if instance_i["question"] == instance_j["question"]:
                        if instance_i["choices"][instance_i["answer"]] == instance_j["choices"][instance_j["answer"]]:
                            train_rqids[i].append(j)
                            train_rqids[j].append(i)
            with open(file_path, "w") as file:
                json.dump(train_rqids, file)

    val_qids = pid_splits[args.val_split]
    test_qids = pid_splits[args.test_split]

    print(f"Number of train problems: {len(train_qids)}")
    print(f"Number of val problems: {len(val_qids)}")
    print(f"Number of test problems: {len(test_qids)}")

    qids = {'train': train_qids, 'val': val_qids, 'test': test_qids}
    if args.r_image == "Yes":
        return problems, qids, name_maps, r_name_maps, image_features, r_image_features, train_rqids
    return problems, qids, name_maps, image_features, train_rqids


class ScienceQADatasetStd(Dataset):
    """Dataset for text-based ScienceQA tasks"""

    def __init__(self, dataset, tokenizer, source_len, target_len, args, test_le=None):
        self.tokenizer = tokenizer
        self.data = dataset
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []

        test_le_data = None
        if test_le is not None:
            test_le_data = json.load(open(test_le))["preds"]

        idx = 0
        for qid in self.data:
            curr_le_data = test_le_data[idx] if test_le_data is not None else None
            if test_le_data is not None:
                idx += 1
            prompt, target = build_train_pair(dataset, qid, args, curr_le_data)
            self.source_text.append(prompt)
            self.target_text.append(target)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = " ".join(str(self.source_text[index]).split())
        target_text = " ".join(str(self.target_text[index]).split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": target["input_ids"].squeeze().tolist(),
        }


class ScienceQADatasetImgNew(Dataset):
    """Dataset for ScienceQA tasks with image features"""

    def __init__(self, problems, qids, name_maps, r_name_maps, tokenizer, source_len, target_len, args,
                 image_features, r_image_features, rqids=None, test_le=None):

        self.rqids = rqids
        self.tokenizer = tokenizer
        self.data = {qid: problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.image_ids = []
        self.r_image_ids = []
        shape = img_shape[args.img_type]
        self.no_img_ids = np.zeros(shape)

        test_le_data = None
        if test_le is not None:
            test_le_data = json.load(open(test_le))["preds"]

        idx = 0
        for qid in self.data:
            curr_le_data = test_le_data[idx] if test_le_data is not None else None
            if test_le_data is not None:
                idx += 1

            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.source_text.append(prompt)
            self.target_text.append(target)

            if str(qid) in name_maps:
                img_index = int(name_maps[str(qid)])
                self.image_ids.append(image_features[img_index])
            else:
                self.image_ids.append(np.zeros(shape))

            if r_name_maps and str(qid) in r_name_maps:
                r_img_index = int(r_name_maps[str(qid)])
                if r_img_index < len(r_image_features):
                    self.r_image_ids.append(r_image_features[r_img_index])
                else:
                    self.r_image_ids.append(np.zeros(shape))
            else:
                self.r_image_ids.append(np.zeros(shape))

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = " ".join(str(self.source_text[index]).split())
        target_text = " ".join(str(self.target_text[index]).split())
        image_ids = torch.tensor(self.image_ids[index]).squeeze()

        if self.rqids is None:
            return {
                "input_ids": self.tokenizer(source_text, return_tensors="pt")["input_ids"].squeeze(),
                "attention_mask": self.tokenizer(source_text, return_tensors="pt")["attention_mask"].squeeze(),
                "image_ids": image_ids,
                "labels": self.tokenizer(target_text, return_tensors="pt")["input_ids"].squeeze(),
            }

        r_image_ids = self.no_img_ids
        r_target_ids = None
        if len(self.rqids[index]) != 0:
            r_index = random.sample(self.rqids[index], 1)[0]
            if hasattr(self, "r_image_ids") and len(self.r_image_ids) > r_index:
                r_image_ids = self.r_image_ids[r_index]
            elif len(self.image_ids) > r_index:
                r_image_ids = self.image_ids[r_index]

            r_target_text = " ".join(str(self.target_text[r_index]).split())
            r_target_ids = self.tokenizer(r_target_text, max_length=self.summ_len,
                                          padding="max_length
