def generate_icf_image(data_item, model, processor, save_dir="./results", alpha=2.0, threshold=0.05):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from qwen_vl_utils import process_vision_info
    import os
    import cv2
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)

    image = data_item["image"]
    prompt = data_item["question"]
    data_id = data_item["id"]
    choices = data_item.get("choices", [])

    if image is None:
        return

    save_subdir = os.path.join(save_dir, str(data_id))
    os.makedirs(save_subdir, exist_ok=True)


    messages_for_solution = [
        {
            "role": "user",
            "content": [
                {"type": "text", "question": prompt},
                {"type": "text", "choices": choices},
            ],
        }
    ]
    text_for_solution = processor.apply_chat_template(
        messages_for_solution, tokenize=False, add_generation_prompt=True
    )
    inputs_for_solution = processor(
        text=[text_for_solution], padding=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs_for_solution = model.generate(
            input_ids=inputs_for_solution["input_ids"],
            attention_mask=inputs_for_solution["attention_mask"],
            max_new_tokens=128,
        )
    solution_generated = processor.decode(outputs_for_solution[0], skip_special_tokens=True)


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "question": prompt},
                {"type": "text", "choices": choices},
                {"type": "text", "answer": solution_generated},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    width_source, height_source = image.size
    width_now, height_now = image_inputs[0].size
    path_width, path_height = width_now // 14, height_now // 14

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    # ------------------ Extract attention ------------------
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    attn_layers = outputs.attentions[-2:]
    attn_stack = torch.stack(attn_layers)
    attn_mean = attn_stack.mean(dim=(0, 1, 2))

    image_token_id = model.config.image_token_id
    input_ids = inputs["input_ids"]
    image_token_mask = (input_ids[0] == image_token_id)
    image_token_indices = image_token_mask.nonzero(as_tuple=True)[0]
    image_attn_matrix = attn_mean[:, image_token_indices]
    phi_avg = image_attn_matrix.mean(dim=0)

    attn_map_merged = phi_avg.reshape(path_height // 2, path_width // 2).unsqueeze(0).unsqueeze(0)
    attn_map_upsampled = F.interpolate(attn_map_merged, size=(path_height, path_width), mode='nearest')
    mask2d = attn_map_upsampled.squeeze(0).squeeze(0)
    masknorm = (mask2d - mask2d.min()) / (mask2d.max() - mask2d.min() + 1e-8)
    maskenhanced = alpha * masknorm

    mask_np = maskenhanced.cpu().numpy()
    masksmooth_np = cv2.GaussianBlur(mask_np, ksize=(3, 3), sigmaX=0.5)
    masksmooth = torch.tensor(masksmooth_np, device=maskenhanced.device)

    maskresized = F.interpolate(
        masksmooth.unsqueeze(0).unsqueeze(0), size=(height_now, width_now), mode='bilinear', align_corners=False
    ).squeeze(0).squeeze(0)
    mask_resized_back = F.interpolate(
        maskresized.unsqueeze(0).unsqueeze(0), size=(height_source, width_source), mode='bilinear', align_corners=False
    ).squeeze(0).squeeze(0)
    mask_resized_back = (mask_resized_back - mask_resized_back.min()) / (
        mask_resized_back.max() - mask_resized_back.min() + 1e-8
    )

    flat_mask = mask_resized_back.flatten()
    k = int(height_source * width_source * 0.9)
    topk_values, topk_indices = torch.topk(flat_mask, k)
    topk_mask_flat = torch.zeros_like(flat_mask)
    topk_mask_flat[topk_indices] = 1.0
    topk_mask = topk_mask_flat.reshape_as(mask_resized_back)

    threshold_mask = (mask_resized_back > threshold).float()
    combined_mask = (topk_mask * threshold_mask).float()

    mask3ch = (1 - combined_mask).unsqueeze(0).repeat(3, 1, 1)
    image_tensor = T.ToTensor()(image).to(mask3ch.device)
    highlight_color = torch.tensor([0.0, 0.0, 0.0], device=mask3ch.device).view(3, 1, 1)
    icf_tensor = image_tensor * (1 - mask3ch) + mask3ch * highlight_color

    icf_image = T.ToPILImage()(icf_tensor.clamp(0, 1).cpu())
    save_path = os.path.join(save_subdir, "image.png")
    icf_image.save(save_path)

    return solution_generated, save_path
