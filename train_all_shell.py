import subprocess

stage0_args = [
    "python", "main.py",
    "--data_root", "./data/",
    "--output_dir", "./results/stage0_sft",
    "--model", "allenai/unifiedqa-t5-base",
    "--options", '["A", "B", "C", "D", "E"]',
    "--epoch", "20",
    "--lr", "5e-5",
    "--bs", "2",
    "--input_len", "512",
    "--output_len", "512",
    "--vot_num", "1",
    "--alpha", "0.5",
    "--eval_bs", "16",
    "--train_split", "train",
    "--val_split", "val",
    "--test_split", "test",
    "--caption_file", "data/captions.json",
    "--img_type", "detr",
    "--prompt_format", "QCM-LE",
    "--user_msg", "stage0_sft",
    "--final_eval",
    "--seed", "42",
    "--forward_vision", "sft"
]

stage1_args = [
    "python", "main.py",
    "--data_root", "./data/",
    "--output_dir", "./results/stage1",
    "--model", ".", 
    "--options", '["A", "B", "C", "D", "E"]',
    "--epoch", "10",
    "--lr", "5e-5",
    "--bs", "2",
    "--input_len", "512",
    "--output_len", "512",
    "--vot_num", "1",
    "--alpha", "0.5",
    "--eval_bs", "16",
    "--train_split", "train",
    "--val_split", "val",
    "--test_split", "test",
    "--caption_file", "data/captions.json",
    "--img_type", "detr",
    "--prompt_format", "QCM-LE",
    "--user_msg", "stage1_rationale",
    "--final_eval",
    "--seed", "42",
    "--forward_vision", "all",
    "--layer_distill", "2",
    "--disstill_alpha", "0.4",
    "--add_alpha", "0.3"
]

stage1_inf_args = [
    "python", "main.py",
    "--data_root", "./data/",
    "--caption_file", "data/captions.json",
    "--model", "./results/stage1/stage1_rationale",
    "--output_dir", "./results/stage1/stage1_rationale",
    "--img_type", "detr",
    "--user_msg", "stage1_rationale",
    "--eval_bs", "2",
    "--input_len", "512",
    "--output_len", "512",
    "--val_split", "val",
    "--test_split", "test",
    "--use_generate",
    "--prompt_format", "QCM-LE",
    "--final_eval",
    "--seed", "42",
    "--evaluate_dir", "./results/stage1/stage1_rationale",
    "--forward_vision", "all"
]

stage2_args = [
    "python", "main.py",
    "--data_root", "./data/",
    "--output_dir", "./results/stage2_answer",
    "--model", "allenai/unifiedqa-t5-base",
    "--options", '["A", "B", "C", "D", "E"]',
    "--epoch", "20",
    "--lr", "5e-5",
    "--bs", "4",
    "--input_len", "512",
    "--output_len", "64",
    "--vot_num", "1",
    "--alpha", "0.5",
    "--eval_bs", "16",
    "--train_split", "train",
    "--val_split", "val",
    "--test_split", "test",
    "--caption_file", "data/captions.json",
    "--img_type", "detr",
    "--prompt_format", "QCMG-A",
    "--eval_le", "./results/stage1/stage1_rationale/predictions_ans_eval.json",
    "--test_le", "./results/stage1/stage1_rationale/predictions_ans_test.json",
    "--user_msg", "stage2_answer",
    "--final_eval",
    "--seed", "42",
    "--forward_vision", "no-layer-distill"
]

stage2_inf_args = [
    "python", "main.py",
    "--data_root", "./data/",
    "--caption_file", "data/captions.json",
    "--model", "./results/stage2_answer/stage2_answer",
    "--output_dir", "./results/stage2_answer",
    "--user_msg", "stage2_rationale_infer",
    "--img_type", "detr",
    "--eval_bs", "4",
    "--input_len", "512",
    "--output_len", "64",
    "--val_split", "val",
    "--test_split", "test",
    "--use_generate",
    "--prompt_format", "QCMG-A",
    "--final_eval",
    "--seed", "42",
    "--eval_le", "./results/stage1/stage1_rationale/predictions_ans_eval.json",
    "--test_le", "./results/stage1/stage1_rationale/predictions_ans_test.json",
    "--evaluate_dir", "./results/stage2_answer/stage2_answer",
    "--forward_vision", "no-layer-distill"
]
if __name__ == "__main__":
    # print("=== Running Stage 0: SFT ===")
    # subprocess.run(stage0_args, check=True)
    print("=== Running Stage 1: Rationale Generation train ===")
    subprocess.run(stage1_args, check=True)
    # print("=== Running Stage 1: Rationale Generation inf===")
    # subprocess.run(stage1_inf_args, check=True)
    # print("\n=== Stage 1 Done. Now Running Stage 2: Answer Train ===")
    # subprocess.run(stage2_args, check=True)
    # print("=== Running Stage 2: answer Generation info ===")
    # subprocess.run(stage2_inf_args, check=True)
    # print("\n=== All Done ===")
