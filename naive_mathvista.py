import argparse
import json
import os
import pickle
import random
import sys
import time
import traceback

import torch
import transformers
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from naive_utils import AutoregressiveSampler, SYSTEM_PROMPT, mcmc_power_samp_ngram
from parse_utils import parse_answer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_results(save_path, save_pkl, results, results_pkl):
    """Atomically save results to avoid corruption on interruption."""
    tmp_json = save_path + ".tmp"
    tmp_pkl = save_pkl + ".tmp"
    
    with open(tmp_json, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_json, save_path)
    
    with open(tmp_pkl, "wb") as f:
        pickle.dump(results_pkl, f)
    os.replace(tmp_pkl, save_pkl)

def load_existing_results(save_path, save_pkl):
    """Load existing results while tolerating missing or corrupt files."""
    results, results_pkl = [], []
    
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        try:
            with open(save_path, "r") as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: failed to load JSON results: {e}")
    
    if os.path.exists(save_pkl) and os.path.getsize(save_pkl) > 0:
        try:
            with open(save_pkl, "rb") as f:
                results_pkl = pickle.load(f)
        except (pickle.UnpicklingError, OSError) as e:
            print(f"Warning: failed to load pickle results: {e}")
    
    return results, results_pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="results/", dest="save_dir")
    parser.add_argument("--model_path", type=str, default="qwen")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--temp", type=float, default=0.25)
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entropy_alpha", type=float, default=1.0)
    parser.add_argument("--scale_alpha", type=float, default=2.0)
    parser.add_argument("--sharpen_alpha", type=float, default=4.0)
    parser.add_argument("--entropy_term", type=str2bool, default=False)
    parser.add_argument("--scale_term", type=str2bool, default=False)
    parser.add_argument("--skip_error", type=str2bool, default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    skip_error = args.skip_error

    # Configure output paths.
    method_str = f"scale_{args.scale_term}_entropy_{args.entropy_term}"
    save_dir = os.path.join(args.save_dir, "mathvista", method_str)
    os.makedirs(save_dir, exist_ok=True)
    
    # Build output file names.
    base_name = f"{args.mcmc_steps}_{args.temp}_{args.batch_idx}_{args.seed}"
    save_path = os.path.join(save_dir, f"{base_name}.json")
    save_pkl = os.path.join(save_dir, f"{base_name}.pkl")
    log_path = os.path.join(save_dir, "logs", f"{base_name}.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Redirect console output to the run log.
    sys.stdout = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout

    # Load the dataset.
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)
    
    # Load the model and processor.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    processor = transformers.AutoProcessor.from_pretrained(
        args.model_path,
    )
    tokenizer.padding_side = "left"

    # Resume from existing results when available.
    results, results_pkl = load_existing_results(save_path, save_pkl)

    if skip_error:
        start_idx = 250 * args.batch_idx + len(results)
        end_idx = min(250 * (args.batch_idx + 1), len(dataset))
    else:
        start_idx = 250 * args.batch_idx
        end_idx = 250 * (args.batch_idx + 1)
        cache_size = len(results)
    
    print(f"Processing batch {args.batch_idx} [{start_idx}/{end_idx}]")
    print(f"Recovered {len(results)} existing results")

    # Process the selected batch in memory.
    interrupted = False
    for data_idx in tqdm(range(start_idx, end_idx), desc="Processing samples"):
        data = {}
        try:
            if not skip_error and (data_idx - start_idx) < cache_size:
                if results[data_idx - start_idx]['status'] == 'success':
                    continue

            data = dataset[data_idx]
            question = data["question"].replace("<image>", "")
            answer = data["answer"]
            # Build the multimodal prompt.
            prompt = question

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data['images'].replace("/media/4/jtb", "/media/data/4/jtb")},
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
            ).to(device)
            
            input_ids = inputs.input_ids
            image_utils = {k: v for k, v in inputs.items() if k != "input_ids"}

            # Run trajectory-guided MCMC sampling.
            start_time = time.time()
            mcmc_output = mcmc_power_samp_ngram(
                autoreg_sampler, 
                input_ids[0].tolist(), 
                image_utils, 
                args.temp, 
                args.mcmc_steps, 
                max_new_tokens=3072, 
                block_num=16,
                entropy_term=args.entropy_term, 
                scale_term=args.scale_term, 
                sharpen_alpha=args.sharpen_alpha,
                entropy_alpha=args.entropy_alpha,
                scale_alpha=args.scale_alpha,
            )
            mcmc_power_samp_output, log_probs_norm, log_probs_unnorm, _, gen_token_cnt = mcmc_output
            exe_time = time.time() - start_time

            # Decode the sampled trajectory.
            mcmc_ids = torch.tensor(mcmc_power_samp_output, device=device).cpu()
            mcmc_completion = tokenizer.decode(mcmc_ids, skip_special_tokens=True)
            if "\nassistant\n" in mcmc_completion:
                mcmc_completion = mcmc_completion.split("\nassistant\n")[1]
            
            # Store results in memory until the batch finishes.
            result_entry = {
                "id": data['index'],
                "question": question,
                "correct_answer": answer,
                "mcmc_completion": mcmc_completion,
                "mcmc_answer": parse_answer(mcmc_completion),
                "status": "success"
            }
            pkl_entry = {
                "id": data['index'],
                "mcmc_ids": mcmc_ids,
                "log_probs_norm": log_probs_norm,
                "log_probs_unnorm": log_probs_unnorm,
                "exe_time": exe_time,
                "token_count": gen_token_cnt
            }

            if not skip_error and (data_idx - start_idx) < cache_size:
                assert results[data_idx - start_idx]['id'] == data['index']
                results[data_idx - start_idx] = result_entry
                results_pkl[data_idx - start_idx] = pkl_entry
            else:
                results.append(result_entry)
                results_pkl.append(pkl_entry)

        except KeyboardInterrupt:
            print("\nProcessing interrupted by the user.")
            interrupted = True
            break
            
        except Exception as e:
            error_msg = f"Sample {data_idx} failed: {e}"
            print(f"\n{error_msg}")
            traceback.print_exc()
            
            # Preserve failed samples so indices remain aligned.
            error_entry = {
                "id": data.get('index', f"unknown_{data_idx}"),
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            error_pkl = {
                "id": data.get('index', f"unknown_{data_idx}"),
                "status": "error",
                "error": str(e)
            }
            
            if not skip_error and (data_idx - start_idx) < cache_size:
                assert results[data_idx - start_idx]['id'] == data['index']
                results[data_idx - start_idx] = error_entry
                results_pkl[data_idx - start_idx] = error_pkl
            else:
                results.append(error_entry)
                results_pkl.append(error_pkl)

    # Save once after the batch finishes or is interrupted.
    try:
        save_results(save_path, save_pkl, results, results_pkl)
        if interrupted:
            print(f"\nPartial results saved to:\n{save_path}\n{save_pkl}")
            sys.exit(1)
        else:
            print(f"\nBatch completed. Results saved to:\n{save_path}\n{save_pkl}")
    except Exception as e:
        print(f"\nFailed to save final results: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout.close()
