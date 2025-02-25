import copy
import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import argparse
import time

from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from distributed_shampoo.examples.trainer_utils import instantiate_optimizer, OptimizerType, GraftingType, DType, PreconditionerComputationType, enum_type_parse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_sst2_file(path):
    examples = []
    with open(path) as f:
        f.readline()
        for line in f:
            sent, label = line.split('\t')
            examples.append([sent, int(label)])
    return examples

class SST2Dataset(Dataset):
    def __init__(self, filename):
        self.examples = read_sst2_file(os.path.join('SST-2', filename))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

def collate_fn(batch, tokenizer):
    sents, labels = zip(*batch)
    labels = torch.FloatTensor(labels)
    encoded = tokenizer(sents, padding=True, return_tensors='pt')
    return encoded['input_ids'], encoded['attention_mask'], labels

def get_init_transformer(transformer):
    def init_transformer(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=transformer.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_transformer

class BertClassifier(nn.Module):
    def __init__(self, drop=0.1):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.score = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(self.encoder.config.hidden_size, 1)
        )
        self.score.apply(get_init_transformer(self.encoder))
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, input_ids, attention_mask, labels):
        hiddens_last = self.encoder(input_ids, attention_mask=attention_mask)[0]
        embs = hiddens_last[:,0,:]
        logits = self.score(embs).squeeze(1)
        loss_total = self.loss(logits, labels)
        return logits, loss_total

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_acc_val(model, dataloader_val, device):
    num_correct_val = 0
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader_val:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits, _ = model(input_ids, attention_mask, labels)
            preds = torch.where(logits > 0., 1, 0)
            num_correct_val += (preds == labels).sum()
    acc_val = num_correct_val / len(dataloader_val.dataset) * 100.
    return acc_val

def configure_optimization(model, num_train_steps, num_warmup_steps, lr, weight_decay=0.01, optimizer_type='adamw', args=None):
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type in ['SGD', 'ADAM', 'DISTRIBUTED_SHAMPOO']:
        optimizer = instantiate_optimizer(
            enum_type_parse(optimizer_type, OptimizerType),
            model,
            lr=lr,
            betas=(args.beta1, args.beta2),
            beta3=args.beta3,
            epsilon=args.epsilon,
            weight_decay=weight_decay,
            momentum=args.momentum,
            dampening=args.dampening,
            max_preconditioner_dim=args.max_preconditioner_dim,
            precondition_frequency=args.precondition_frequency,
            start_preconditioning_step=args.start_preconditioning_step,
            inv_root_override=args.inv_root_override,
            exponent_multiplier=args.exponent_multiplier,
            use_nesterov=args.use_nesterov,
            use_bias_correction=args.use_bias_correction,
            use_decoupled_weight_decay=args.use_decoupled_weight_decay,
            grafting_type=enum_type_parse(args.grafting_type, GraftingType),
            grafting_epsilon=args.grafting_epsilon,
            grafting_beta2=args.grafting_beta2,
            use_merge_dims=args.use_merge_dims,
            distributed_config=None,
            preconditioner_dtype=DType.FP32,
            preconditioner_computation_type=PreconditionerComputationType.EIGEN_ROOT_INV,
        )
    else:
        raise ValueError('Invalid optimizer type: {}'.format(optimizer_type))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
    return optimizer, scheduler

def compute_loss(model, dataset, tokenizer, batch_size, device='cuda'):
    model = model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                          collate_fn=lambda b: collate_fn(b, tokenizer))
    total_loss = 0.
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            _, loss_batch = model(input_ids, attention_mask, labels)
            total_loss += loss_batch.item()
    avg_loss = total_loss / len(dataset)
    return avg_loss

def train(model, dataset_train, dataset_val, tokenizer, batch_size=32, batch_size_eval=1024, num_warmup_steps=10, 
          lr=0.00005, num_epochs=3, clip=1., verbose=True, device='cuda', warmup_ratio=0.01, log_interval=100, weight_decay=0.01, 
          optimizer_type='adamw', args=None):
    model = model.to(device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2, 
                                collate_fn=lambda b: collate_fn(b, tokenizer))
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size_eval, shuffle=False, num_workers=2,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    
    num_train_steps = len(dataset_train) // batch_size * num_epochs
    if warmup_ratio != -1:
        num_warmup_steps = int(num_train_steps * warmup_ratio)
        
    optimizer, scheduler = configure_optimization(model, num_train_steps, num_warmup_steps, lr, weight_decay=0.01, optimizer_type=optimizer_type, args=args)

    loss_avg = float('inf')
    acc_train = 0.
    best_acc_val = 0.
    for epoch in range(num_epochs):
        model.train()
        loss_total = 0.
        num_correct_train = 0
        interval_loss = 0.
        interval_count = 0
        for batch_ind, (input_ids, attention_mask, labels) in enumerate(dataloader_train):
            input_ids = input_ids.to(device).long()
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits, loss_batch_total = model(input_ids, attention_mask, labels)
            preds = torch.where(logits > 0., 1, 0)
            num_correct_train += (preds == labels).sum()
            loss_total += loss_batch_total.item()
            interval_loss += loss_batch_total.item()
            interval_count += input_ids.size(0)

            if (batch_ind + 1) % log_interval == 0:
                interval_avg_loss = interval_loss / interval_count
                print(batch_ind + 1, '/', len(dataloader_train), 'batches done | interval avg loss {:.4f}'.format(interval_avg_loss))
                if torch.isnan(torch.tensor(interval_avg_loss)):
                    print("Loss is NaN, stopping training")
                    exit(1)
                interval_loss = 0.
                interval_count = 0

            loss_batch_avg = loss_batch_total / input_ids.size(0)
            loss_batch_avg.backward()

            if clip > 0.:
                nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_avg = loss_total / len(dataloader_train.dataset)
        acc_train = num_correct_train / len(dataloader_train.dataset) * 100.

        acc_val = get_acc_val(model, dataloader_val, device)

        if verbose:
            print('Epoch {:3d} | avg loss {:8.4f} | train acc {:2.2f} | val acc {:2.2f}'.format(epoch + 1, loss_avg, acc_train, acc_val))

        if acc_val > best_acc_val:
            best_acc_val = acc_val

    if verbose:
        print('Final avg loss {:8.4f} | final train acc {:2.2f} | best val acc {:2.2f}'.format(loss_avg, acc_train, best_acc_val))

    return best_acc_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--batch_size_eval', type=int, default=1024, help='Batch size for evaluation')
    parser.add_argument('--num_warmup_steps', type=int, default=10, help='Number of warmup steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.01, help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--optimizer_type', type=str, default='adamw', help='Optimizer type to use')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--quiet', action='store_true', help='Suppress training progress output')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--log_interval', type=int, default=100, help='How often to log training progress')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for optimizer')
    parser.add_argument('--beta3', type=float, default=-1.0, help='Beta3 for optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-12, help='Epsilon for optimizer')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum for optimizer')
    parser.add_argument('--dampening', type=float, default=0.0, help='Dampening for optimizer')
    parser.add_argument('--max_preconditioner_dim', type=int, default=1024, help='Max preconditioner dimension')
    parser.add_argument('--precondition_frequency', type=int, default=100, help='Precondition frequency')
    parser.add_argument('--start_preconditioning_step', type=int, default=-1, help='Start preconditioning step')
    parser.add_argument('--inv_root_override', type=int, default=0, help='Inverse root override')
    parser.add_argument('--exponent_multiplier', type=float, default=1.0, help='Exponent multiplier')
    parser.add_argument('--use_nesterov', action='store_true', help='Use Nesterov momentum')
    parser.add_argument('--use_bias_correction', action='store_true', default=True, help='Use bias correction')
    parser.add_argument('--use_decoupled_weight_decay', action='store_true', default=True, help='Use decoupled weight decay')
    parser.add_argument('--grafting_epsilon', type=float, default=1e-8, help='Grafting epsilon')
    parser.add_argument('--grafting_beta2', type=float, default=0.999, help='Grafting beta2')
    parser.add_argument('--use_merge_dims', action='store_true', default=True, help='Use merge dimensions')
    parser.add_argument('--grafting_type', type=str, default='ADAM', help='Grafting type')
    args = parser.parse_args()
    print(args)

    set_seed(42)
  
    dataset_train = SST2Dataset('train.tsv')
    dataset_val = SST2Dataset('dev.tsv')
    print('{} train sents, {} val sents'.format(len(dataset_train), len(dataset_val)))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('The pretrained tokenizer in bert-base-uncased has vocab size {:d}\n'.format(tokenizer.vocab_size))

    model = BertClassifier()
    print('Model has {} parameters\n'.format(count_params(model)))

    start_time = time.time()
    best_acc_val = train(model,
                        dataset_train,
                        dataset_val,
                        tokenizer,
                        batch_size=args.batch_size,
                        batch_size_eval=args.batch_size_eval,
                        num_warmup_steps=args.num_warmup_steps,
                        optimizer_type=args.optimizer_type,
                        lr=args.lr,
                        num_epochs=args.num_epochs,
                        clip=args.clip,
                        verbose=not args.quiet,
                        device=args.device,
                        warmup_ratio=args.warmup_ratio,
                        log_interval=args.log_interval,
                        weight_decay=args.weight_decay,
                        args=args)
    train_time = (time.time() - start_time) / 60
    print('Train time: {:.2f} minutes'.format(train_time))

    loss_train = compute_loss(model, dataset_train, tokenizer, args.batch_size, args.device)
    loss_val = compute_loss(model, dataset_val, tokenizer, args.batch_size_eval, args.device)
    print('Train loss: {:.4f}, Val loss: {:.4f}'.format(loss_train, loss_val))