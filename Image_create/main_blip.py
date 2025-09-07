import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from datadeal import build_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
train_data, val_data, test_data = build_dataset()

image_path = "path/to/your/image.jpg"
image = train_data[0]["image"]


prompt = "Describe the contents of the image in detail."
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)


outputs = model.generate(**inputs, output_attentions=True, return_dict_in_generate=True)
tokens = outputs.sequences
caption = processor.decode(tokens[0], skip_special_tokens=True)
print("desible：", caption)

cross_attentions = outputs.cross_attentions


last_token_cross = [layer_attn[0] for layer_attn in cross_attentions[-1]]
attn_tensor = torch.stack(last_token_cross, dim=0)

focused_attn = attn_tensor[-3:].mean(dim=0)

mask_values = focused_attn.mean(dim=(1, 2, 3)).squeeze(0)

mask_values = mask_values[1:] if mask_values.shape[0] > (image.size[1]//16)*(image.size[0]//16) else mask_values

patchh = patchw = int(np.sqrt(mask_values.shape[0]))
mask_2d = mask_values.reshape(patchh, patchw).cpu().numpy()

mask_norm = (mask_2d - mask_2d.min()) / (mask_2d.max() - mask_2d.min() + 1e-8)
alpha = 1.0
mask_enhanced = alpha * mask_norm


kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
mask_tensor = torch.tensor(mask_enhanced, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
mask_smooth = F.conv2d(mask_tensor, kernel, padding=1)
mask_smooth = mask_smooth.squeeze().cpu().numpy()


H, W = image.size[1], image.size[0]
mask_resized = Image.fromarray(mask_smooth).resize((W, H), resample=Image.BILINEAR)
mask_resized = np.array(mask_resized, dtype=np.float32)
mask_resized = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)
mask_resized = np.clip(mask_resized, 0, 1)


img_np = np.array(image).astype(np.float32) / 255.0

I_cf = img_np * (1 - mask_resized[..., None]) + 1.0 * mask_resized[..., None]


I_cf = (I_cf * 255).astype(np.uint8)
result_img = Image.fromarray(I_cf)
result_img.save("counterfactual_image.png")
print("counterfactual_image.png. ready")
