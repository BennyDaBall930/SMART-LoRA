"""
Data Processing Pipeline - Minimax Version
=========================================
Loads and preprocesses datasets for instruction tuning.
Sanitized version for public release.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer


class ZEngineerDataset(Dataset):
    """Dataset for instruction tuning training."""
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 768,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # ChatML special tokens
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _format_chatml(self, item: Dict[str, str]) -> str:
        """Format item as ChatML conversation."""
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        # Build user message
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction
        
        # Format as ChatML
        formatted = (
            f"{self.im_start}user\n{user_content}{self.im_end}\n"
            f"{self.im_start}assistant\n{output_text}{self.im_end}"
        )
        return formatted
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        formatted_text = self._format_chatml(item)
        
        # Tokenize
        encodings = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # For causal LM, labels = input_ids (shifted internally by the model)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels (set to -100)
        labels[attention_mask == 0] = -100
        
        # Mask instruction tokens (only train on assistant response)
        # Find where assistant response starts
        assistant_start_text = f"{self.im_start}assistant\n"
        assistant_start_ids = self.tokenizer.encode(
            assistant_start_text, add_special_tokens=False
        )
        
        # Find the position of assistant start in input_ids
        input_ids_list = input_ids.tolist()
        assistant_start_pos = None
        
        for i in range(len(input_ids_list) - len(assistant_start_ids) + 1):
            if input_ids_list[i:i + len(assistant_start_ids)] == assistant_start_ids:
                assistant_start_pos = i + len(assistant_start_ids)
                break
        
        # Mask everything before assistant response
        if assistant_start_pos is not None:
            labels[:assistant_start_pos] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_splits(
    data: List[Dict[str, str]],
    train_ratio: float = 0.95,
    seed: int = 42,
) -> tuple[List[Dict], List[Dict]]:
    """Split data into train and validation sets."""
    import random
    random.seed(seed)
    
    # Shuffle
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    # Split
    split_idx = int(len(data) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    
    return train_data, val_data


def get_dataloaders(
    config,
    tokenizer: PreTrainedTokenizer,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    # Load data
    print(f"Loading dataset from {config.paths.dataset_path}...")
    all_data = load_jsonl(config.paths.dataset_path)
    print(f"Loaded {len(all_data)} samples")
    
    # Split
    train_data, val_data = create_splits(
        all_data,
        train_ratio=config.data.train_split,
        seed=config.data.seed,
    )
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    
    # Create datasets
    train_dataset = ZEngineerDataset(
        train_data, tokenizer, config.model.max_length
    )
    val_dataset = ZEngineerDataset(
        val_data, tokenizer, config.model.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.training.dataloader_pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.per_device_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.training.dataloader_pin_memory,
    )
    
    return train_loader, val_loader


def test_data_processing(config):
    """Test the data processing pipeline."""
    print("=" * 60)
    print("Testing Data Processing Pipeline")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {config.paths.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.paths.model_dir,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Load a few samples
    all_data = load_jsonl(config.paths.dataset_path)
    sample = all_data[0]
    
    print(f"\n--- Sample Entry ---")
    print(f"Instruction: {sample.get('instruction', '')[:100]}...")
    print(f"Input: {sample.get('input', '')[:100]}...")
    print(f"Output: {sample.get('output', '')[:100]}...")
    
    # Test dataset class
    dataset = ZEngineerDataset([sample], tokenizer, config.model.max_length)
    processed = dataset[0]
    
    print(f"\n--- Tokenized ---")
    print(f"Input IDs shape: {processed['input_ids'].shape}")
    print(f"Attention mask shape: {processed['attention_mask'].shape}")
    print(f"Labels shape: {processed['labels'].shape}")
    
    # Count non-masked labels
    non_masked = (processed['labels'] != -100).sum().item()
    total = processed['labels'].shape[0]
    print(f"Training tokens: {non_masked}/{total} ({100*non_masked/total:.1f}%)")
    
    # Analyze token distribution
    print(f"\n--- Token Length Distribution (first 100 samples) ---")
    lengths = []
    for item in all_data[:100]:
        ds = ZEngineerDataset([item], tokenizer, 2048)  # Use large max for analysis
        proc = ds[0]
        actual_len = (proc['attention_mask'] == 1).sum().item()
        lengths.append(actual_len)
    
    print(f"Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")
    
    print("\n✓ Data processing test completed successfully!")


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "config"))
    from config_minimax import get_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run data processing test")
    args = parser.parse_args()
    
    config = get_config()
    
    if args.test:
        test_data_processing(config)
