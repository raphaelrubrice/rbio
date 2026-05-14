# # Training Rbio (A Demo)
# In this script we will demonstrate how to train Rbio on perturbation data obtained from
# the PertQA dataset available originally published here https://github.com/genentech/PerturbQA and adapted to our use case.
#
# Rbio implements LLM post-training using soft-verification mechanisms so that knowledge from biology models such as a virtual cell model (VCM) can be distilled and used within the LLM, rather than relying on hard ground truth labels obtained experimentally which are usually scarce and often costly.
#
# In this example, we use a simplified "VCM" consisting of a Multi Layer Perceptron (MLP) trained to answer perturbation questions. It exposes an interface that returns a probability when prompted with two gene names. This is the probability that a knockout of gene_a is having an effect on the expression of gene_b.
#
# We use this signal as a soft verification signal within our reward mechanism in order to post-train our LLM. This improves the LLM capabilities to answer questions of the form "Is a knockdown of <gene_a> in <cell_line> cells likely to result in differential expression of gene_b?"

# ## Imports, global variables, random seeds

import os
import re
from typing import List

import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from trl import GRPOTrainer

from rewards import *

from utils import (
    set_random_seeds,
    load_mlp_classifier,
    setup_model_and_tokenizer,
    create_training_config,
    mlp_classifier_inference
)

from dual_utils import load_kg_data, add_dual, build_dual_prompt

# Disabling logging
os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"

# Training configuration
MODEL_NAME = "Qwen/Qwen3-1.7B"
N_STEPS = 100000
BATCH_SIZE = 4
NUM_GENERATIONS = 4
SAVE_EVERY = 10000
OUTPUT_DIR = "./checkpoints"

# Global step counter
STEP_COUNT = 0

# MLP classifier configuration (mlp was not trained on k562 cells)
MLP_MODEL_PATH = "./mlp_model.pt"
EMBEDDING_FILE = "./esm_embedding_dictionary_filled.pkl"

# Dataset paths
DATASET_PATHS = [
    "./k562-train-v0.3.0.csv",
]

THINK_BRIEF = "\nThink step by step, but be concise. Keep your reasoning under 100 words."

# Set seeds globally
set_random_seeds(42)


# # Simplified VCM
# Our simplified virtual cell model (VCM) is a MLP Classifier. This serves the purpose of a soft labeler for our reward strategy, as it returns probabilities of gene pairs being differentially expressed.

# In[ ]:


class MLPClassifier(nn.Module):
    """Simple MLP classifier for gene pair classification"""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.model(x)
        return result


# Global variables for MLP model and embeddings
mlp_model = None
embeddings_dict = None


# # Dataset
#
# `load_and_prepare_dataset` has the purpose of loading the training datasets (k562 only in this example) as a pandas dataframe.
#
# `create_mlp_labeled_dataset_generator` yields samples from this dataset that have been soft-labeled by our simplified VCM: the MLP defined above.

# In[ ]:


def load_and_prepare_dataset(dataset_paths: List[str], balance_pos_neg: bool = True) -> pd.DataFrame:
    """Load CSV datasets and combine them into a single DataFrame"""
    if len(dataset_paths) == 1:
        dataset_df = pd.read_csv(dataset_paths[0])
    else:
        dataset_list = []
        for path in dataset_paths:
            dataset_list.append(pd.read_csv(path))
        dataset_df = pd.concat(dataset_list, ignore_index=True)

    print(f"Loaded dataset with {len(dataset_df)} rows")
    return dataset_df


def create_mlp_labeled_dataset_generator(dataset_df: pd.DataFrame, tokenizer, balance_pos_neg: bool = True):
    """Generate training examples with MLP-based labeling"""
    if balance_pos_neg:
        # Use 2x the dataset length to ensure enough samples for training
        dataset_length = len(dataset_df) * 2
    else:
        dataset_length = len(dataset_df)

    for i in range(dataset_length):
        # Sample from dataset (with replacement for longer training)
        sample_idx = i % len(dataset_df)
        row = dataset_df.iloc[sample_idx]

        # Prepare sample data for MLP classification
        sample_data = {
            "system_prompt": row["system_prompt"] + THINK_BRIEF,
            "user_prompt": row["user_prompt"],
            "keywords": row["keywords"]
        }

        # Get MLP prediction
        mlp_probability = mlp_classifier_inference(sample_data)

        # Determine label based on MLP probability
        predicted_label = 1 if mlp_probability > 0.5 else 0

        # Prepare sample with MLP-generated label
        sample = {
            "system_prompt": row["system_prompt"],
            "user_prompt": row["user_prompt"],
            "label": predicted_label,
            "classes": "no|yes",
            "class_confidences": f"{1.0-mlp_probability:.3f}|{mlp_probability:.3f}",
            "keywords": row["keywords"],
            "task": row["task"],
            "mlp_probability": mlp_probability
        }

        # Format messages for chat template
        messages = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["user_prompt"]},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )

        yield {
            "prompt":                    prompt,
            "label":                     sample["label"],
            "classes":                   sample["classes"],
            "class_confidences":         sample["class_confidences"],
            "keywords":                  sample["keywords"],
            "task":                      sample["task"],
            "system_prompt":             sample["system_prompt"],
            "user_prompt":               sample["user_prompt"],
            "gene_perturbed":            row["gene_perturbed"],
            "gene_monitored":            row["gene_monitored"],
            "perturbed_gene_summary":    row["perturbed_gene_summary"],
            "gene_monitored_rn_summaries": row["gene_monitored_rn_summaries"],
            "potential_genes":           row["potential_genes"],
        }

def load_and_prepare_dataset(dataset_paths: List[str], balance_pos_neg: bool = True) -> pd.DataFrame:
    """Load CSV datasets and combine them into a single DataFrame"""
    if len(dataset_paths) == 1:
        dataset_df = pd.read_csv(dataset_paths[0])
    else:
        dataset_list = []
        for path in dataset_paths:
            dataset_list.append(pd.read_csv(path))
        dataset_df = pd.concat(dataset_list, ignore_index=True)

    print(f"Loaded dataset with {len(dataset_df)} rows")
    return dataset_df


def create_dual_dataset_generator(dataset_df: pd.DataFrame, tokenizer, balance_pos_neg: bool = True):
    """Generate training examples with dual tasks"""
    if balance_pos_neg:
        # Use 2x the dataset length to ensure enough samples for training
        dataset_length = len(dataset_df) * 2
    else:
        dataset_length = len(dataset_df)

    for i in range(dataset_length):
        # Sample from dataset (with replacement for longer training)
        sample_idx = i % len(dataset_df)
        row = dataset_df.iloc[sample_idx]

        # Prepare sample data for MLP classification
        sample_data = {
            "system_prompt": row["system_prompt"] + THINK_BRIEF,
            "user_prompt": row["user_prompt"],
            "keywords": row["keywords"]
        }

        # Get MLP prediction
        mlp_probability = mlp_classifier_inference(sample_data)

        dual_prompt = make_dual(row["gene_perturbed"], row["gene_monitored"])
        
        # Determine label based on MLP probability
        predicted_label = 1 if mlp_probability > 0.5 else 0

        # Prepare sample with MLP-generated label
        sample = {
            "system_prompt": row["system_prompt"],
            "user_prompt": row["user_prompt"],
            "label": predicted_label,
            "classes": "no|yes",
            "class_confidences": f"{1.0-mlp_probability:.3f}|{mlp_probability:.3f}",
            "keywords": row["keywords"],
            "task": row["task"],
            "mlp_probability": mlp_probability
        }

        # Format messages for chat template
        messages = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["user_prompt"]},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        yield {
            "prompt": prompt,
            "label": sample["label"],
            "classes": sample["classes"],
            "class_confidences": sample["class_confidences"],
            "keywords": sample["keywords"],
            "task": sample["task"],
            "system_prompt": sample["system_prompt"],
            "user_prompt": sample["user_prompt"],
            "dual_prompt": sample["dual_prompt"]
        }


# # Rewards
# `reward_answer_against_label` rewards the answer provided by the model (typically yes/no according to our prompts) by assigning the probability of the selected answer as estimated by the simplified VCM soft verifier.
#
# `composite_formatting_reward` makes sure formatting of the LLM output is compliant to our expectations and guidelines expressed in system prompt.
#
# `keywords_mentioned_in_think` makes sure specific keywords (typically gene names) are mentioned during reasoning.
#
# `compute_simple_reward` is used by the trainer to assign a reward to a generated trace.


def make_rollout_fn(model, tokenizer, prompt_to_dual):
    """Return a rollout_func that generates first completions, then dual completions."""
    def rollout(prompts, trainer):
        import torch.nn.functional as F
        n_gen = trainer.args.num_generations

        # First generation
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                num_return_sequences=n_gen,
                return_dict_in_generate=True,
                output_scores=True,
            )

        prompt_len     = enc["input_ids"].shape[1]
        completion_ids = out.sequences[:, prompt_len:]
        first_texts    = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Per-token log-probs
        logprobs = torch.stack(
            [F.log_softmax(s, dim=-1) for s in out.scores], dim=1
        )
        token_logprobs = logprobs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)

        prompt_ids_rep = enc["input_ids"].repeat_interleave(n_gen, dim=0)

        # Build dual prompts conditioned on each first completion's answer
        prompts_rep = [p for p in prompts for _ in range(n_gen)]
        dual_prompts, gene_monitored_list, potential_genes_list = [], [], []
        for p, first_text in zip(prompts_rep, first_texts):
            dual = prompt_to_dual.get(p, {})
            answer = extract_binary_answer(first_text)
            answer_str = "yes" if answer is True else "no"
            dual_prompts.append(build_dual_prompt(
                dual.get("gene_perturbed", ""),
                answer_str,
                dual.get("perturbed_gene_summary", ""),
                dual.get("gene_monitored_rn_summaries", ""),
                dual.get("potential_genes", ""),
            ))
            gene_monitored_list.append(dual.get("gene_monitored", ""))
            potential_genes_list.append(dual.get("potential_genes", ""))

        # Second generation — format via chat template with thinking enabled
        dual_chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": dp}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for dp in dual_prompts
        ]
        enc2 = tokenizer(
            dual_chat_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        with torch.no_grad():
            out2 = model.generate(
                **enc2, max_new_tokens=512,
                do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
            )
        second_texts = tokenizer.batch_decode(
            out2[:, enc2["input_ids"].shape[1]:], skip_special_tokens=True
        )

        return {
            "prompt_ids":           prompt_ids_rep,
            "completion_ids":       completion_ids,
            "logprobs":             token_logprobs,
            "second_completions":   second_texts,
            "gene_monitored_list":  gene_monitored_list,
            "potential_genes_list": potential_genes_list,
        }
    return rollout


def reward_answer_against_label(completion: str, classes: str, class_confidence: str) -> float:
    """Compute reward based on whether answer matches expected label"""
    answer = extract_binary_answer(completion)
    if answer is None:
        return 0.0

    answer = "yes" if answer else "no"
    possible_classes = classes.split("|")
    confidences = [float(c) for c in class_confidence.split("|")]

    for label, conf in zip(possible_classes, confidences):
        if answer == label.strip().lower():
            return conf

    return 0.0


def composite_formatting_reward(text: str, use_go: bool = False) -> float:
    """Compute composite formatting reward based on multiple checks"""
    at_least_one_think = has_at_least_one_think(text)
    has_tags = has_any_tag(text)

    checks = [
        at_least_one_think,
        low_untagged_ratio(text),
        is_not_too_long(text),
        has_one_answer(text),
        answer_after_thinks(text),
        thinks_have_text(text) * at_least_one_think,
        no_nested_tags(text) * has_tags,
        has_limited_thinks(text) * at_least_one_think,
        all_tags_properly_closed(text) * has_tags,
        ends_with_answer(text),
        starts_with_think(text),
    ]

    # Remove start_with_think dependency if using GO ontology
    if use_go:
        checks = checks[:-1]

    return sum(checks) / len(checks)

def keywords_mentioned_in_think(text: str, keywords: str) -> float:
    """Check how many keywords are mentioned in think sections"""
    keyword_list = [k for k in keywords.split("|") if k]

    if not keyword_list:
        return 1.0

    think_contents = extract_think(text)
    if not think_contents:
        return 0.0

    found_keywords = 0
    for keyword in keyword_list:
        if keyword in think_contents:
            found_keywords += 1

    return found_keywords / len(keyword_list)


def compute_simple_reward(
    completions: List[str],
    label: List[int],
    classes: List[str],
    class_confidences: List[str],
    keywords: List[str],
    second_completions: List[str] = None,
    gene_monitored_list: List[str] = None,
    potential_genes_list: List[str] = None,
    **kwargs
) -> List[float]:
    """Compute rewards for model completions using format, mention, answer, and dual rewards."""
    scores = []

    global STEP_COUNT

    for i, (completion, lbl, class_list, confidences, keyword_list) in enumerate(zip(
        completions, label, classes, class_confidences, keywords
    )):
        format_reward  = composite_formatting_reward(completion, use_go=False)
        mention_reward = keywords_mentioned_in_think(completion, keyword_list)
        answer_reward  = reward_answer_against_label(completion, class_list, confidences)

        dual_reward = 0.0
        dual_format_reward = 0.0
        dual_mention_reward = 0.0
        candidate_adherence = 0.0

        if second_completions and gene_monitored_list:
            gene_m = gene_monitored_list[i].upper()
            second = second_completions[i]

            dual_format_reward = composite_formatting_reward(second)
            dual_mention_reward = keywords_mentioned_in_think(second, gene_m)

            answer_match = re.search(r'<answer>(.*?)</answer>', second, re.DOTALL | re.IGNORECASE)
            # Only reward if gene appears inside a proper <answer> tag, not anywhere in the text
            dual_reward = 1.0 if (answer_match and gene_m in answer_match.group(1).strip().upper()) else 0.0

            if potential_genes_list and i < len(potential_genes_list):
                candidates = {g.upper() for g in potential_genes_list[i].split("|") if g}
                answer_text = answer_match.group(1).strip().upper() if answer_match else second.strip().upper()
                if any(c in answer_text for c in candidates):
                    candidate_adherence = 0.15

        total_score = (format_reward + 2.0 * answer_reward + mention_reward
                       + dual_format_reward + dual_mention_reward + dual_reward
                       + candidate_adherence)
        scores.append(total_score)

        if STEP_COUNT % 100 == 0:
            print("\n" + "="*80)
            print(f"DEBUG: Sample {STEP_COUNT} - Step {STEP_COUNT}")
            print("="*80)

            print(f"MODEL OUTPUT:")
            print(f"{completion}")
            print()

            if second_completions:
                print(f"DUAL OUTPUT (gene_monitored={gene_monitored_list[i] if gene_monitored_list else '?'}):")
                print(f"{second_completions[i]}")
                print()

            print(f"REWARD BREAKDOWN:")
            print(f"  Format reward:        {format_reward:.3f}")
            print(f"  Mention reward:       {mention_reward:.3f}")
            print(f"  Answer reward:        {answer_reward:.3f}")
            print(f"  Dual format reward:   {dual_format_reward:.3f}")
            print(f"  Dual mention reward:  {dual_mention_reward:.3f}")
            print(f"  Dual accuracy reward: {dual_reward:.3f}")
            print(f"  Candidate adherence:  {candidate_adherence:.3f}")
            print(f"  Total score:          {total_score:.3f}")
            print()

            print(f"EXPECTED vs PREDICTED:")
            print(f"  VCM binarized label: {lbl}")
            print(f"  Possible classes: {class_list}")
            print(f"  Label VCM confidences: {confidences}")
            print(f"  Keywords: {keyword_list}")
            print()

            print(f"REWARD DETAILS:")
            print(f"  Answer extraction: {extract_binary_answer(completion)}")
            print(f"  Think content: {extract_think(completion)[:100]}...")
            print("="*80 + "\n")
    STEP_COUNT += 1

    return scores


# # Training


print("Starting RBIO training with streaming MLP labeling...")

# Load and prepare dataset
print("Loading dataset...")
dataset_df = load_and_prepare_dataset(DATASET_PATHS)

# Load MLP classifier
print("Loading MLP classifier...")
load_mlp_classifier(MLP_MODEL_PATH, EMBEDDING_FILE, MLPClassifier)

# Load KG data once (gene summaries + STRING KG)
print("Loading KG data for dual task...")
gs, kg = load_kg_data()

# Enrich dataset with dual task columns
print("Adding dual task columns...")
dataset_df = add_dual(dataset_df, gs, kg)

# Setup model and tokenizer
model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)

# Build dataset eagerly so we can construct the prompt→dual lookup for rollout_func
print("Building dataset...")
dataset_records = list(create_mlp_labeled_dataset_generator(
    dataset_df=dataset_df, tokenizer=tokenizer, balance_pos_neg=True
))
dataset = Dataset.from_list(dataset_records)

_DUAL_FIELDS = [
    "gene_perturbed", "gene_monitored",
    "perturbed_gene_summary", "gene_monitored_rn_summaries", "potential_genes",
]
prompt_to_dual = {r["prompt"]: {k: r[k] for k in _DUAL_FIELDS} for r in dataset_records}

# Create training configuration
print("Setting up training configuration...")
training_config = create_training_config(
    output_dir=OUTPUT_DIR,
    batch_size=BATCH_SIZE,
    num_generations=NUM_GENERATIONS,
    max_steps=N_STEPS,
    save_every=SAVE_EVERY
)

# Create trainer
print("Creating GRPO trainer...")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=compute_simple_reward,
    args=training_config,
    train_dataset=dataset,
    rollout_func=make_rollout_fn(model, tokenizer, prompt_to_dual),
)

# Start training
print(f"Starting training for {N_STEPS} steps...")
trainer.train()

print("Training completed!")


# ## Notes
# - This code is a different implementation compared to the code that has been used to train the methods discussed in our paper "Rbio: ...."
# - If you are interested only in using the perturbation data we employ in this dataset, please refer to the original repository https://github.com/genentech/PerturbQA and cite the work from our colleagues at Genentech accordingly
