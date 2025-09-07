import os
import json
from PIL import Image
import os
import json
from PIL import Image

class ScienceQADataset:
    def __init__(self, json_path, image_root_dir, max_samples=None):

        self.json_path = json_path
        self.image_root_dir = image_root_dir
        self.max_samples = max_samples
        self.data = self._load_dataset()

    def _load_dataset(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            problems = json.load(f)

        data_list = []
        count = 0

        for pid, pdata in problems.items():
            if self.max_samples and count >= self.max_samples:
                break

            split = pdata.get("split", "train")
            question = pdata.get("question", "")

            image_name = None
            if isinstance(pdata.get("image"), str) and pdata.get("image"):
                image_name = pdata["image"]
            elif isinstance(pdata.get("choice_0"), str) and pdata.get("choice_0"):
                image_name = pdata["choice_0"]
            elif isinstance(pdata.get("choice_1"), str) and pdata.get("choice_1"):
                image_name = pdata["choice_1"]
            elif isinstance(pdata.get("choice_2"), str) and pdata.get("choice_2"):
                image_name = pdata["choice_2"]
            elif isinstance(pdata.get("choice_3"), str) and pdata.get("choice_3"):
                image_name = pdata["choice_3"]
            elif isinstance(pdata.get("choice_4"), str) and pdata.get("choice_4"):
                image_name = pdata["choice_4"]
            if image_name is None:
                image = None
            else:
                image_path = os.path.join(self.image_root_dir, str(split), str(pid), image_name)
                if os.path.exists(image_path):
                    try:
                        image = Image.open(image_path).convert("RGB")
                    except Exception as e:
                        print(f"[{pid}] on open：{image_path} faile：{e}")
                        image = None
                else:
                    print(f"[{pid}] no exist：{image_path}")
                    image = None

            data_item = {
                "id": int(pid),
                "split": split,
                "question": question,
                "image": image,
                "solution": pdata.get("solution", "")
            }

            data_list.append(data_item)
            count += 1

        return data_list

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_all(self):
        return self.data



def build_dataset():
    json_path = ""
    image_root_dir =""

    dataset = ScienceQADataset(json_path, image_root_dir)
    print("all", len(dataset))

    return dataset




class M3CoTDataset:
    def __init__(self, json_path, image_root_dir, max_samples=None):

        self.json_path = json_path
        self.image_root_dir = image_root_dir
        self.max_samples = max_samples
        self.data = self._load_dataset()

    def _load_dataset(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            problems = json.load(f)

        data_list = []
        count = 0

        for pid, pdata in problems.items():
            if self.max_samples and count >= self.max_samples:
                break

            question = pdata.get("question", "")
            solution = pdata.get("solution", "")
            choices= pdata.get("choices", [])
            image_path = os.path.join(self.image_root_dir, f"{pid}.png")

            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"[{pid}] no open：{image_path} faile：{e}")
                    image = None
            else:
                print(f"[{pid}] no exist：{image_path}")
                image = None

            data_item = {
                "id": pid,
                "question": question,
                "solution": solution,
                "image": image,
                "choices": choices,
            }

            data_list.append(data_item)
            count += 1

        return data_list

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_all(self):
        return self.data

def build_m3dataset():
    json_path = r""
    image_root_dir = r""

    dataset = M3CoTDataset(json_path, image_root_dir)
    print("all：", len(dataset))

    return dataset

build_m3dataset()
