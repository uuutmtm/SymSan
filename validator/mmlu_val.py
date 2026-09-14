"""Utility evaluation: MMLU accuracy, perplexity,
and logit changes."""
import torch
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
import logging
from typing import Optional, Dict, Tuple

# Set up simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MMLUTester:
    def __init__(self, model, tokenizer, data_dir="datasets/mmlu/val", device="cuda"):
        """
        Initialize the MMLU Tester.
        
        Args:
            model: The transformer model to evaluate.
            tokenizer: The tokenizer for the model.
            data_dir: Path to the MMLU dataset directory (defaults to val set).
            device: Device to run evaluation on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.device = device
        
        # Ensure model is in eval mode
        self.model.eval()
        # self.model.to(self.device) # model might be on multiple devices via accelerate
        
        # Handle pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Choices mapping
        self.choices = ["A", "B", "C", "D"]
        # Pre-calculate choice IDs for efficiency
        # Using [0] or [-1] depends on tokenizer, usually single token expected
        self.choice_ids = []
        for c in self.choices:
            token_ids = self.tokenizer.encode(c, add_special_tokens=False)
            self.choice_ids.append(token_ids[-1]) # Take the last token ID if multiple
        self.choice_ids = torch.tensor(self.choice_ids).to(self.model.device)

    def format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(chr(65 + j), df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def evaluate(self, ref_model: Optional[torch.nn.Module] = None, num_samples: Optional[int] = None, batch_size: int = 8) -> Dict[str, float]:
        """
        Run MMLU evaluation and calculate metrics.
        
        Args:
            ref_model: (Optional) Reference model to calculate logit changes against.
            num_samples: Number of samples per subject to evaluate (None for all).
            batch_size: Batch size for inference.
            
        Returns:
            Dict containing: 'mmlu_acc', 'ppl', 'logit_mean_change', 'logit_max_change'
        """
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_dir}")
            return {"mmlu_acc": 0.0, "ppl": 0.0, "logit_mean_change": 0.0, "logit_max_change": 0.0}

        if ref_model:
            ref_model.eval()
            # ref_model.to(self.device)

        total_correct = 0
        total_count = 0
        total_loss = 0.0
        total_tokens = 0
        
        logit_diff_sum = 0.0
        logit_diff_count = 0
        logit_max_diff = 0.0
        
        logger.info(f"Starting evaluation on {len(csv_files)} subjects. Batch size: {batch_size}")
        
        for csv_file in tqdm(csv_files, desc="Eval Subjects"):
            df = pd.read_csv(csv_file, header=None)
            limit = len(df)
            if num_samples is not None and num_samples > 0:
                limit = min(num_samples, limit)
            
            prompts = []
            labels = []
            
            # Prepare data
            for i in range(limit):
                prompts.append(self.format_example(df, i, include_answer=False))
                labels.append(df.iloc[i, df.shape[1] - 1])
            
            # Batch loop
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                batch_labels = labels[i : i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                
                with torch.no_grad():
                    # 1. Forward pass current model
                    # Compute loss for PPL (using inputs as labels, ignoring pad)
                    # Create labels: -100 for pad tokens
                    batch_labels_tensor = inputs.input_ids.clone()
                    if self.tokenizer.pad_token_id is not None:
                         batch_labels_tensor[batch_labels_tensor == self.tokenizer.pad_token_id] = -100
                    
                    outputs = self.model(**inputs, labels=batch_labels_tensor)
                    logits = outputs.logits
                    loss = outputs.loss
                    
                    # Update PPL stats
                    # loss is mean reduction by default, we need total loss to average correctly over all batches?
                    # Or just average the batch losses. Weighted average is better.
                    # Approx: Accumulate loss * batch_size (or valid tokens), then divide.
                    # Here we simplify: accumulate loss * batch_size, assume similar token counts or just average batch losses.
                    # Better: PPL = exp(total_nll / total_tokens)
                    # Huggingface loss is NLL per token.
                    # We can roughly use the batch average loss.
                    # PPL calculation here is approximate based on the questions.
                    
                    # Accumulate separate counts for PPL
                    current_loss = loss.item()
                    # Just keep average of batch losses
                    total_loss += current_loss * len(batch_prompts)
                    
                    # 2. MMLU Accuracy
                    # Get last token logits
                    # Assuming right padding. Last real token index = sum(mask) - 1
                    last_token_indices = inputs.attention_mask.sum(1) - 1
                    
                    # Select logits: [batch, vocab]
                    selected_logits = logits[torch.arange(logits.size(0)), last_token_indices]
                    
                    # Choice probabilities
                    choice_logits = selected_logits[:, self.choice_ids]
                    pred_indices = torch.argmax(choice_logits, dim=1).cpu()
                    
                    for j, pred_idx in enumerate(pred_indices):
                        if self.choices[pred_idx] == batch_labels[j]:
                            total_correct += 1
                    
                    total_count += len(batch_prompts)
                    
                    # 3. Logit Change (if ref_model provided)
                    if ref_model is not None:
                        ref_outputs = ref_model(**inputs)
                        ref_logits = ref_outputs.logits
                        
                        # Calculate difference (L1 or L2 or Abs)
                        # The user asked for "mean change" and "max change". 
                        # Usually |new - old|.
                        diff = (logits - ref_logits).abs()
                        
                        # Update stats
                        batch_mean = diff.mean().item()
                        batch_max = diff.max().item()
                        
                        logit_diff_sum += batch_mean * logits.numel() # weighted by number of elements
                        logit_diff_count += logits.numel()
                        
                        if batch_max > logit_max_diff:
                            logit_max_diff = batch_max

        # Finalize Metrics
        acc = total_correct / total_count if total_count > 0 else 0.0
        avg_loss = total_loss / total_count if total_count > 0 else 0.0
        ppl = np.exp(avg_loss)
        
        logit_mean_change = 0.0
        if logit_diff_count > 0:
            logit_mean_change = logit_diff_sum / logit_diff_count
            
        results = {
            "mmlu_acc": acc,
            "ppl": ppl,
            "logit_mean_change": logit_mean_change,
            "logit_max_change": logit_max_diff
        }
        
        logger.info(f"Evaluation Results: {results}")
        return results

# Helper function to use directly
def test_mmlu(model, tokenizer, ref_model=None, data_dir="datasets/mmlu/data/val", device="cuda", num_samples=None):
    tester = MMLUTester(model, tokenizer, data_dir, device)
    return tester.evaluate(ref_model=ref_model, num_samples=num_samples)
