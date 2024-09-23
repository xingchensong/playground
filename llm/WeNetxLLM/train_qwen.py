import os
import argparse
import torch
import torch.distributed as dist
import datasets

from torch.nn.utils import clip_grad_norm_
from transformers import Qwen2Config, Qwen2Tokenizer, Qwen2ForCausalLM
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


config = {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  # "max_window_layers": 24,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 4,  # 24
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  # "sliding_window": 32768,
  "tie_word_embeddings": True,
  "torch_dtype": "float32",  # "bfloat16"
  "transformers_version": "4.43.3",  # "4.40.1"
  "use_cache": True,
  "use_mrope": False,
  "use_sliding_window": False,  # True
  "vocab_size": 151936,
  "attn_implementation": "sdpa",  # eager
}


class LLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Qwen2ForCausalLM(Qwen2Config(**config))  # random init
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, labels):
        result = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return result


class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length,
                                  truncation=True, padding='max_length')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        if input_ids.numel() == 0:
            return self.__getitem__((idx + 1) % len(self.dataset))
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def collate_fn(batch):
    ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    print('Train on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def get_args():
    parser = argparse.ArgumentParser(description='train qwen on wiki2')
    parser.add_argument('--device', required=True, type=str, choices=["cuda", "cpu"], help='device for train')
    parser.add_argument('--output_dir', required=True, type=str, help='dir to save result')
    parser.add_argument('--batch_size', required=True, type=int, help='batch size (per-device) for train')
    parser.add_argument('--num_workers', type=int, default=4, help='workers for dataloader')
    parser.add_argument('--prefetch', type=int, default=5, help='prefetch for dataloader')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda":
        assert (torch.cuda.is_available())
        world_size, local_rank, rank = init_distributed()
    else:
        world_size, local_rank, rank = 1, 0, 0

    device = torch.device(args.device)
    model = LLM().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    train_dataset = TextDataset(
        datasets.load_dataset("wikitext", "wikitext-2-v1")['train'],
        tokenizer
    )
    cv_dataset = TextDataset(
        datasets.load_dataset("wikitext", "wikitext-2-v1")['validation'],
        tokenizer
    )
    if args.device == "cuda":
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                  shuffle=(sampler is None), num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch, collate_fn=collate_fn)
    cv_dataloader = DataLoader(cv_dataset, batch_size=args.batch_size, sampler=None,
                               shuffle=False, num_workers=args.num_workers,
                               prefetch_factor=args.prefetch, collate_fn=collate_fn)

    for epoch in range(1):
        model.train()
        if sampler:
            sampler.set_epoch(epoch)
        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids, attention_mask, labels).loss
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
            optimizer.zero_grad()

        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(cv_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                loss = model(input_ids, attention_mask, labels).loss
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
        avg_loss = total_loss / total_samples
        if rank == 0:
            print(f"Epoch {epoch + 1}, Validation Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
