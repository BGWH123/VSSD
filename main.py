import os
import numpy as np
import torch
import re
import json
import argparse
import random
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    T5ForConditionalGeneration
from model import T5ForConditionalGeneration, T5ForMultimodalGenerationVSSDCoT
from model_new import T5ForMultimodalGenerationVSSDCoTNew, T5ForConditionalGeneration, T5Config
from model_sft import T5ForMultimodalGenerationVSSDCoTSFT
from utils_data import img_shape, load_data_std, load_data_img, ScienceQADatasetStd, ScienceQADatasetImg,ScienceQADatasetImgNew
from utils_evaluate import get_scores, get_scores_m3
# from datasets import load_metric
import evaluate
from tqdm import tqdm

from rich import box
from rich.table import Column, Table
from rich.console import Console


console = Console(record=True)
import nltk
from torch.utils.data import DataLoader


# Set the current working directory
def extract_ans(ans):
    pattern = re.compile(r'The answer is \(([A-Z])\)')
    # pattern = re.compile(r'\(([A-Z])\)')
    res = pattern.findall(ans)

    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"
    return answer


import torch


def check_model_weights(model, threshold=1e-8):
    print("🔍 Checking model weights...")
    issues = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 

        with torch.no_grad():
            if torch.isnan(param).any():
                print(f"[ NaN] Found in: {name}")
                issues.append((name, "NaN"))
            elif torch.all(param == 0):
                print(f"[ All Zeros] Found in: {name}")
                issues.append((name, "All Zeros"))
            elif torch.max(torch.abs(param)) < threshold:
                print(f"[ Very Small] Found in: {name}, max={param.abs().max().item():.2e}")
                issues.append((name, "Very Small"))

    if not issues:
        print(" All model weights look fine.")
    else:
        print(f" Found {len(issues)} potential issues in model parameters.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--vot_num', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train',
                        choices=['train', 'trainval', 'minitrain', 'tinytrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival', 'tinyval', "dev"])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest', 'tinytest'])
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='The path to the checkpoint to resume training from.')

    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', default="True",
                        help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')  # 保存目录
    parser.add_argument('--img_type', type=str, default='detr', choices=['detr', 'clip', 'resnet'],
                        help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-LE', help='prompt format template',
                        choices=['QCM-A', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE', 'QCMG-A', 'QCM-E'])  # 选择的模板
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--forward_vision', type=str, default="all", choices=["sft", "all", "no-layer-distill", "test"])
    parser.add_argument('--layer_distill', type=int, default=11, help='the number of layers to distill vision features')
    parser.add_argument('--disstill_alpha', type=float, default=0.2, help='the scaling factor for distillation loss')
    parser.add_argument('--add_alpha', type=float, default=0.1, help='steering loss scaling factor')
    parser.add_argument('--r_image', type=str, default="No", choices=["Yes", "No"])
    args = parser.parse_args()
    return args


def T5Trainer(
        dataframe, args,
):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True  

    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = T5Tokenizer.from_pretrained(args.model)  # google/flan-t5-xl

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    problems = dataframe['problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']
    train_rqids = dataframe['rqids']
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/", "-")
        gpu_count = torch.cuda.device_count()
        # save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
        save_dir = f"{args.output_dir}/{args.user_msg}"
        print("save_dir", save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    padding_idx = tokenizer._convert_token_to_id(tokenizer.pad_token)  
    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
        if args.resume_from_checkpoint:
            console.log(f"[Model]: Resuming from checkpoint {args.resume_from_checkpoint}...\n")
            model = T5ForMultimodalGenerationVSSDCoT.from_pretrained(args.resume_from_checkpoint, patch_size=patch_size,
                                                                     padding_idx=padding_idx, save_dir=save_dir,
                                                                     vot_num=args.vot_num, alpha=args.alpha)
        else:
            import torch.nn as nn
            console.log(f"[Model]: Loading {args.model}...\n")
            if args.forward_vision == "all":
                model = T5ForMultimodalGenerationVSSDCoTNew.from_pretrained(
                    args.model,
                    patch_size=patch_size,
                    padding_idx=padding_idx,
                    save_dir=save_dir,
                    vot_num=args.vot_num,
                    alpha=args.alpha,
                    layer_distill=args.layer_distill,
                    disstill_alpha=args.disstill_alpha,  
                    add_alpha=args.add_alpha,  # steering
                )

            elif args.forward_vision == "test":
                model = T5ForMultimodalGenerationVSSDCoT.from_pretrained(
                    args.model,
                    patch_size=patch_size,
                    padding_idx=padding_idx,
                    save_dir=save_dir,
                    vot_num=args.vot_num,
                    alpha=args.alpha,
                )

            elif args.forward_vision == "sft":
                model = T5ForMultimodalGenerationVSSDCoTSFT(
                    config=T5Config.from_pretrained(args.model),
                    patch_size=patch_size,
                    padding_idx=padding_idx,
                    save_dir=save_dir,
                    vot_num=args.vot_num,
                    alpha=args.alpha
                )
                state_dict = torch.load(os.path.join(args.model, 'pytorch_model.bin'))
                model.load_state_dict(state_dict, strict=False)

                model.image_dense = nn.Linear(model.patch_dim, model.model_dim)
                model.gate_dense = nn.Linear(2 * model.hidden_size, model.hidden_size)
                model.mha_layer = nn.MultiheadAttention(embed_dim=model.hidden_size, kdim=model.hidden_size,
                                                        vdim=model.hidden_size, num_heads=1, batch_first=True)
            elif args.forward_vision == "no-layer-distill":
                model = T5ForMultimodalGenerationVSSDCoT(
                    config=T5Config.from_pretrained(args.model),
                    patch_size=patch_size,
                    padding_idx=padding_idx,
                    save_dir=save_dir,
                    vot_num=args.vot_num,
                    alpha=args.alpha
                )
                state_dict = torch.load(os.path.join(args.model, 'pytorch_model.bin'))
                model.load_state_dict(state_dict, strict=False)
                model.image_dense = nn.Linear(model.patch_dim, model.model_dim)
                model.gate_dense = nn.Linear(2 * model.hidden_size, model.hidden_size)
                model.mha_layer = nn.MultiheadAttention(embed_dim=model.hidden_size, kdim=model.hidden_size,
                                                        vdim=model.hidden_size, num_heads=1, batch_first=True)


            # print(model.image_dense.state_dict())

            check_model_weights(model)
        if args.evaluate_dir is not None:
            print("Inference Phase")
        name_maps = dataframe['name_maps']
        if args.r_image == "Yes":
            r_name_maps = dataframe['r_name_maps']
            r_image_features = dataframe['r_image_features']
        image_features = dataframe['image_features']

        # print(train_rqids)
        # train_rqids=None  # none!!!!!!!
        if args.r_image=="Yes":
            print("Using r_rationale image features")
            # train_rqids = dataframe['rqids']
            train_set = ScienceQADatasetImgNew(
                problems,
                train_qids,
                name_maps,
                r_name_maps,
                tokenizer,
                args.input_len,
                args.output_len,
                args,
                image_features,
                r_image_features=r_image_features,
                rqids=train_rqids,
            )
        else:
            print("Using original image features")
            train_set = ScienceQADatasetImg(
                problems,
                train_qids,
                name_maps,
                tokenizer,
                args.input_len,
                args.output_len,
                args,
                image_features,
                rqids=train_rqids,
            )
        # train_set=train_set[:10000]
        eval_set = ScienceQADatasetImg(
            problems,
            val_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            test_le=args.eval_le,
        )
        test_set = ScienceQADatasetImg(
            problems,
            test_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            test_le=args.test_le,
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model)
        train_set = ScienceQADatasetStd(
            problems,
            train_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
        )
        eval_set = ScienceQADatasetStd(
            problems,
            val_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.eval_le,
        )
        test_set = ScienceQADatasetStd(
            problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.test_le,
        )

    datacollator = DataCollatorForSeq2Seq(tokenizer)

    # accuracy for answer inference
    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct += 1
        return {'accuracy': 1.0 * correct / len(targets)}

    # metric = load_metric("rouge")
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels): 
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_rougel(eval_preds):  
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="no",
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            report_to="none",
        )

    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="accuracy" if args.prompt_format != "QCM-LE" else "rougeL",
            predict_with_generate=args.use_generate,
            load_best_model_at_end=True,
            report_to="none",
        )
    # # print(model)
    model.set_conf(False)  # zghed
    if args.prompt_format == 'QCM-LE' and args.evaluate_dir is None:  # froze LM's encoder
        model.set_conf(True)
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.shared.weight.requires_grad = True
    trainer = VSSDCoTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_acc if args.prompt_format != "QCM-LE" else compute_metrics_rougel
    )

    if args.evaluate_dir is None:
        trainer.train()
        trainer.save_model(save_dir)
    if args.evaluate_dir:
        output_json_path = os.path.join(save_dir, "predictions_ans_test.json")
        trainer.predict_in_batches(test_dataset=test_set, max_length=args.output_len, output_json_path=output_json_path,
                                   batch_size=args.eval_bs, tokenizer=tokenizer, args=args, test_qids=test_qids)

        # generate the rationale for the eval set
        if args.prompt_format == "QCM-LE":
            torch.cuda.empty_cache()
            trainer.generate_rationale_for_eval_set(eval_set=eval_set, args=args, tokenizer=tokenizer,
                                                    save_dir=save_dir)


class VSSDCoTTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_in_batches(self, test_dataset, max_length, output_json_path, batch_size=16, tokenizer=None, args=None,
                           test_qids=None):
        self.args.max_length = max_length
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        results_ans = {}
        results_rationale = {}
        results_reference = {}
        preds_list = []
        targets_list = []
        num_fail = 0
        num_batches = len(dataloader)

        pbar = tqdm(total=num_batches, desc="Predicting", ncols=100)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.args.device)
            labels = batch['labels'].to(self.args.device)
            attention_mask = batch["attention_mask"].to(self.args.device)
            image_ids = batch["image_ids"].to(dtype=torch.float).to(self.args.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, image_ids=image_ids, attention_mask=attention_mask,
                                     labels=labels)

            preds = outputs.logits.argmax(dim=-1)
            preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            targets = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            actual_batch_size = len(preds)
            for idx, qid in enumerate(range(batch_idx * batch_size, batch_idx * batch_size + actual_batch_size)):
                pred = preds[idx]
                ref = targets[idx]
                actual_qid = test_qids[qid]
                extract_pred = extract_ans(pred)

                if extract_pred != "FAILED":
                    if extract_pred in args.options:
                        extract_pred = args.options.index(extract_pred)
                    else:
                        extract_pred = random.choice(range(0, len(args.options)))
                else:
                    num_fail += 1
                    extract_pred = random.choice(range(len(args.options)))

                results_ans[str(actual_qid)] = extract_pred
                results_rationale[str(actual_qid)] = pred
                results_reference[str(actual_qid)] = ref
                preds_list.append(pred)
                targets_list.append(ref)

            pbar.update(1)

        pbar.close()

        output_data = {
            "num_fail": num_fail,
            "results_ans": results_ans,
            "results_rationale": results_rationale,
            "results_reference": results_reference
        }

        if args.val_split == "dev":
            scores = get_scores_m3(results_ans, results_rationale, results_reference,
                                   "./data/data_m3/scienceqa/problems.json")
        else:
            scores = get_scores(results_ans, results_rationale, results_reference, "./data/scienceqa/problems.json")
        for key, value in scores.items():
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")

        output_data["scores"] = scores
        output_data["preds"] = preds_list
        output_data["labels"] = targets_list
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)

    def generate_rationale_for_eval_set(self, eval_set, args, tokenizer, save_dir):
        self.args.max_length = args.output_len
        dataloader = DataLoader(eval_set, batch_size=args.eval_bs, shuffle=False)

        preds_list = []
        targets_list = []

        num_batches = len(dataloader)
        pbar = tqdm(total=num_batches)
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.args.device)
            labels = batch['labels'].to(self.args.device)
            attention_mask = batch["attention_mask"].to(self.args.device)
            image_ids = batch["image_ids"].to(dtype=torch.float).to(self.args.device)
            no_img_ids = None
            # shape = image_ids.shape #cd
            # no_img_ids = np.zeros(shape) #cd
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, r_image_ids=no_img_ids, image_ids=image_ids,
                                     attention_mask=attention_mask, labels=labels)

            preds = outputs.logits.argmax(dim=-1)
            preds_decoded = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            targets_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)

            preds_list.extend(preds_decoded)
            targets_list.extend(targets_decoded)

            pbar.update()

        output_data = {
            "preds": preds_list,
            "labels": targets_list
        }

        output_prediction_file = os.path.join(save_dir, "predictions_ans_eval.json")
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))


if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(  
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    args = parse_args()
    print("args", args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.img_type is not None:
        if args.r_image=="Yes":
            print("Using r_rationale image features")
            problems, qids, name_maps,r_name_maps, image_features, r_image_features,train_rqids = load_data_img(
                args)  # probelms, test question ids, shot example ids
            dataframe = {'problems': problems, 'qids': qids, 'name_maps': name_maps, 'r_name_maps': r_name_maps,'image_features': image_features,'r_image_features': r_image_features,
                         'rqids': train_rqids}
        else:
            problems, qids, name_maps, image_features, train_rqids = load_data_img(
                args)  # probelms, test question ids, shot example ids
            dataframe = {'problems': problems, 'qids': qids, 'name_maps': name_maps, 'image_features': image_features,
                         'rqids': train_rqids}
    else:
        problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems': problems, 'qids': qids}

    T5Trainer(
        dataframe=dataframe,
        args=args
    )
