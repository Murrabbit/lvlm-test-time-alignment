import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from linguistic_utils import length_reward, repeat_ratio


SYSTEM_PROMPT = (
    "You are a helpful assistant! Please first think about the reasoning process "
    "as an internal monologue and then provide the final answer for the user's "
    "question within \\boxed{}."
)

def seed_everything(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, torch, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    
def compute_entropies_from_logits(unscaled_logits, chunk_size=128):
    """Compute token entropies in chunks to limit peak GPU memory usage.

    Args:
        unscaled_logits: Tensor shaped ``[seq_len, 1, vocab_size]`` or
            ``[seq_len, vocab_size]``.
        chunk_size: Number of token positions processed per chunk.

    Returns:
        A list of ``seq_len`` entropy values stored on the CPU.
    """
    if unscaled_logits.dim() == 3:
        assert unscaled_logits.shape[1] == 1, "Only batch_size=1 supported"
        unscaled_logits = unscaled_logits.squeeze(1)
    
    seq_len = unscaled_logits.shape[0]
    entropies = torch.empty(seq_len, device=unscaled_logits.device)
    
    # Process logits in chunks to avoid materializing a full probability matrix.
    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)
        chunk = unscaled_logits[start_idx:end_idx]
        
        log_probs = F.log_softmax(chunk, dim=-1)
        probs = torch.exp(log_probs)
        
        # Compute one position at a time to keep temporary allocations small.
        chunk_entropies = torch.empty(end_idx - start_idx, device=chunk.device)
        for i in range(chunk.shape[0]):
            chunk_entropies[i] = -torch.sum(probs[i] * log_probs[i])
        
        entropies[start_idx:end_idx] = chunk_entropies
    
    return entropies.cpu().tolist()

class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


def naive_temp(p : AutoregressiveSampler, context, image_utils, temp, seq_len, scale_alpha, sharpen_alpha):
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)

    if tokenizer.pad_token_id is not None:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
    else:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    for k, v in image_utils.items():
        if k not in model_inputs:
            model_inputs[k] = v

    output = p.model.generate(
        **model_inputs,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]

    E_list = compute_entropies_from_logits(unscaled_logits)

    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (sharpen_alpha * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm, log_probs_unnorm, E_list


def naive_temp_scale(p : AutoregressiveSampler, context, image_utils, temp, seq_len, scale_alpha, sharpen_alpha):
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    # Build an attention mask aligned with the input sequence.
    if tokenizer.pad_token_id is not None:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
    else:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    input_ids = torch.cat([input_ids, input_ids], dim=0)
    attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    for k, v in image_utils.items():
        if k not in model_inputs:
            model_inputs[k] = torch.cat([v, v], dim=0)

    output = p.model.generate(
        **model_inputs,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )

    logits_w_img = tuple(logit_tensor[0:1] for logit_tensor in output.logits)
    scores_w_img = tuple(score_tensor[0:1] for score_tensor in output.scores)
    logits_wo_img = tuple(logit_tensor[1:2] for logit_tensor in output.logits)


    unscaled_logits = torch.stack(logits_w_img, dim=0)
    scaled_logits = torch.stack(scores_w_img, dim=0)
    unscaled_logits_wo_img = torch.stack(logits_wo_img, dim=0)

    E_list = compute_entropies_from_logits(unscaled_logits)

    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0] == len(E_list)

    idx = tokens.view(unscaled_logits.shape[0], 1, 1)
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    # Smaller scale factors impose a stronger image-grounding requirement.
    log_probs_raw = torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx).view(-1)
    log_probs_raw_wo_img = torch.gather(F.log_softmax(unscaled_logits_wo_img, dim=-1), -1, idx).view(-1)  

    alpha = scale_alpha
    margin = 1e-3

    smooth_eps = 0.01
    log_ratios = torch.log((torch.exp(log_probs_raw) + smooth_eps) / (torch.exp(log_probs_raw_wo_img) + smooth_eps))

    # Calibrate the adjustment to zero when the probability ratio is one.
    calibration = alpha * F.softplus(torch.tensor(-margin, device=log_ratios.device))

    adjustment = -alpha * F.softplus(-log_ratios - margin) + calibration

    # Penalize only tokens that become less likely when the image is present.
    mask_ratio = (log_ratios < 0).float()
    adjustment = adjustment * mask_ratio  

    # Apply the visual-grounding adjustment before sharpening.
    log_probs_unnorm_correct = ((log_probs_raw + adjustment) * sharpen_alpha).tolist()

    return prop, log_probs_norm, log_probs_unnorm_correct, E_list


def mcmc_power_samp_ngram(
    p : AutoregressiveSampler, 
    context, 
    image_utils, 
    temp, 
    mcmc_steps, 
    max_new_tokens, 
    block_num=16, 
    entropy_term=False, 
    scale_term=False, 
    entropy_alpha=1.0,
    scale_alpha=2.0,
    sharpen_alpha=4.0
):

    if scale_term:
        samp_func = naive_temp_scale
    else:
        samp_func = naive_temp

    c = len(context)
    gen = context.copy()
    img = image_utils.copy()

    log_probs_norm = []
    log_probs_unnorm = []
    TE_total = []
    eos = p.tokenizer.eos_token_id


    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    attempts = 0
    acceptances = 0
    gen_token_cnt = 0
    latest_accept = None

    for block_idx in range(block_num):
        if block_idx >= 10:
            return latest_accept

        l_ori = len(gen)
        gen, lp_norm, lp_unnorm, TE = samp_func(
            p,
            gen,
            img,
            temp=temp,
            seq_len=jump_size + len(gen),
            scale_alpha=scale_alpha,
            sharpen_alpha=sharpen_alpha,
        )

        if scale_term:
            gen_token_cnt += (len(gen) - l_ori) * 2
        else:
            gen_token_cnt += (len(gen) - l_ori)

        # The proposal distribution uses temperature-scaled probabilities.
        log_probs_norm.extend(lp_norm)
        # The target distribution uses sharpened probabilities.
        log_probs_unnorm.extend(lp_unnorm)
        TE_total.extend(TE)

        for _ in range(mcmc_steps):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # Query the model for a new suffix proposal.
            prop, log_prob_prop, target_log_prob_prop, TE_prop = samp_func(p, gen[:idx], img, temp=temp, seq_len=t, scale_alpha=scale_alpha,sharpen_alpha=sharpen_alpha)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            assert(len(TE_prop) == s - idx) 

            if scale_term:
                gen_token_cnt += (s - idx) * 2
            else:
                gen_token_cnt += (s - idx)

            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]

            assert len(target_log_prob_cur) == s - idx
            
            TE_cur = TE_total.copy()[idx-c:s-c]

            # Compute the base Metropolis-Hastings acceptance ratio.
            log_r = (
                sum(target_log_prob_prop) + sum(log_prob_cur)
                - sum(target_log_prob_cur) - sum(log_prob_prop)
            )
            
            if entropy_term:
                entropy_penalty = entropy_alpha * (-sum(TE_prop) + sum(TE_cur))
                log_r += entropy_penalty
            
            cur_repeat_ratio = repeat_ratio(gen[c:])
            prop_repeat_ratio = repeat_ratio(prop[c:])

            cur_len  = max(1, t - c)
            prop_len = max(1, s - c)
            cur_score  = cur_len  / (1.0 + math.log(cur_len) * (cur_repeat_ratio))
            prop_score = prop_len / (1.0 + math.log(prop_len) * (prop_repeat_ratio))
            cur_length_reward = length_reward(cur_score, L=128)
            prop_length_reward = length_reward(prop_score, L=128)

            length_penalty = prop_length_reward - cur_length_reward
            log_r += length_penalty

            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()
                TE_total[idx-c:] = TE_prop.copy()

                num_eos = gen.count(eos)
    
                if num_eos >= 3:
                    eos_idx = [i for i, t in enumerate(gen) if t == eos][2]
                    gen = gen[:eos_idx + 1]
                    log_probs_norm = log_probs_norm[:eos_idx + 1]
                    log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
                    acceptance_ratio = acceptances/attempts if attempts else 0
                    latest_accept =  (gen, log_probs_norm, log_probs_unnorm, acceptance_ratio, gen_token_cnt)

        num_eos = gen.count(eos)
    
        if num_eos >= 3:
            eos_idx = [i for i, t in enumerate(gen) if t == eos][2]
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts  if attempts else 0
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio, gen_token_cnt

    acceptance_ratio = acceptances/attempts  if attempts else 0
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio, gen_token_cnt