import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from tqdm import tqdm
from datadeal import build_dataset,build_m3dataset
import os
from model import generate_icf_image

save_dir = ""
os.makedirs(save_dir, exist_ok=True)
model_path = r""


print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)

model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
data= build_m3dataset()
processor = AutoProcessor.from_pretrained(model_path)


for data_item in tqdm(data):
    print(f"Processing ID: {data_item['id']}")
    generate_icf_image(data_item, model, processor, save_dir=save_dir, alpha=2.0, threshold=0.05)
