import os

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse
from qwen_vl_utils import process_vision_info
import sys
import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets
from parse_utils import parse_answer
import pickle
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers

from constants import *
from blink_utils import *
import traceback

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def formatted_prompt_mmstar(text):
    text = text.replace("<image>", "")
    # return f"{text}\nPlease select the correct answer from the choices above."
    return f"{text}\nChoose the single best answer from the choices above."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", action = "store", type = str, default = "results/",  dest = "save_dir")
    parser.add_argument("--model_path", action = "store", default = "qwen", type = str)
    parser.add_argument("--dataset_path", action = "store", default = "", type = str)
    parser.add_argument("--temperature", action = "store", default = 0.25, type = float, dest = "temperature")
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 10)
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    parser.add_argument("--entropy_term", type=str2bool, default=False)
    parser.add_argument("--scale_term", type=str2bool, default=False)
    args = parser.parse_args()

    random.seed(args.seed)

    model_path = args.model_path
    temp = args.temperature
    mcmc_steps = args.mcmc_steps
    entropy_term = args.entropy_term
    scale_term = args.scale_term
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    method_str = f"scale_{scale_term}_entropy_{entropy_term}"
    print(method_str)

    save_dir = os.path.join(args.save_dir, "mmstar")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, method_str, str(mcmc_steps) + "_" + str(temp) + "_" + str(args.batch_idx)  + "_" + str(args.seed) + ".json")
    save_pkl = save_path.replace(".json", ".pkl")

    os.makedirs(os.path.join(save_dir, method_str, "logs"), exist_ok=True)
    log_path = os.path.join(save_dir, method_str, "logs", str(mcmc_steps) + "_" + str(temp) + "_" + str(args.batch_idx)  + "_" + str(args.seed) + ".txt")
    sys.stdout = open(log_path, "w", encoding="utf-8")

    json_file = args.dataset_path
    dataset = json.load(open(json_file, "r"))

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)
    hf_model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code = True)
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)
    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        # min_pixels=256*28*28, 
        # max_pixels=2048*28*28
    )

    tokenizer.padding_side = "left"
    results = []
    results_pkl = []

    start = 250 * args.batch_idx
    end = 250 * (args.batch_idx+1)

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)
        start += len(results)

        with open(save_pkl, "rb") as f:
            results_pkl = pickle.load(f)

    try:
        for problem, data in tqdm(enumerate(dataset[start:end]), desc = "Benchmark on MMStar", total=len(dataset[start:end])):
            question = formatted_prompt_mmstar(data["question"])
            # print(question)
            answer = data["answer"]
            actions = data['actions']

            # if actions[0] == "Task Initialization" and actions[1] == "Visual Detection" and "Logical Reasoning" not in actions:
            #     actions = actions[1:]

            print("============ PROMPT ============")
            prompt = PROMPT.format(
                question=question,
                hint = " ".join([f"{i}. {a}" for i,a in enumerate(actions, start=1)])
            )
            print(prompt)
            messages = [
                {
                    "role": "system",
                    "content": BLINK_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data['image_path'].replace("/media/4/jtb","/root/autodl-tmp")},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            input_ids = inputs.input_ids
            image_utils = inputs.copy()
            image_utils.pop("input_ids")

            prefx = [idx.item() for idx in input_ids[0]]

            start_time = time.time()
            mcmc_power_samp_output, log_probs_norm, log_probs_unnorm, acceptance_ratio, gen_token_cnt = mcmc_power_samp_blink_ignore(
                autoreg_sampler, 
                prefx, 
                image_utils, 
                temp, 
                mcmc_steps, 
                max_new_tokens=3072, 
                entropy_term=entropy_term, 
                scale_term=scale_term, 
                blink_block=2
            )
            end_time = time.time()
            exe_time = end_time - start_time

            mcmc_power_samp_ids = torch.tensor([mcmc_power_samp_output], dtype=torch.long, device=device).squeeze().to("cpu")
            mcmc_completion = tokenizer.decode(mcmc_power_samp_ids, skip_special_tokens=True).split("\nassistant\n")[1]
            
            print("BLINK DONE")
            print(question)
            print(mcmc_completion)
            print("Token 消耗总计:", gen_token_cnt)
            print(answer)

            mcmc_answer = parse_answer(mcmc_completion)

            results.append({
                "id": data['id'],
                "question": question,
                "correct_answer": answer,
                "mcmc_completion": mcmc_completion,
                "mcmc_answer": mcmc_answer
            })

            results_pkl.append({
                "id": data['id'],
                "mcmc_power_samp_ids": mcmc_power_samp_ids,
                "log_probs_norm": log_probs_norm,
                "log_probs_unnorm": log_probs_unnorm,
                "exe_time": exe_time
            })

    except KeyboardInterrupt:
        print("\n\n⚠️ User interrupted processing. Saving partial results...")
    except Exception as e:
        print(f"\n\n🔥 UNEXPECTED ERROR: {str(e)}")
        traceback.print_exc()
    finally:

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        with open(save_pkl, "wb") as f:
            pickle.dump(results_pkl, f)











        













