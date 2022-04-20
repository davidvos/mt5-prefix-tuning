from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

import os
import json
import copy

from benchmark_reader import Benchmark, select_files
class WebNLG:

    def __init__(self, tokenizer, raw_path='../data/release_v3.0/ru', language='en', data_path='../data/preprocessed', split='train'):
        
        if not os.path.exists(f'{data_path}/{split}.json'):
            b = Benchmark()
            files = select_files(raw_path)
            b.fill_benchmark(files)
            b.b2json(data_path, f'{split}.json')
        
        with open(f'{data_path}/{split}.json', 'r') as f:
            dataset = json.load(f)
            entries = dataset['entries']

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []
        for i, entry in enumerate(entries):
            sents = entry[str(i + 1)]['lexicalisations']
            triples = entry[str(i + 1)]['modifiedtripleset']
            
            rela_lst = []
            temp_triples = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                rela_lst.append(rela)
                temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in sents:
                if sent["lang"] == language:
                    full_tgt_lst.append(sent["lex"])
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(rela_lst)
                    if split == 'dev':
                        break
            
        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)

        self.examples = []
        self.targets = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            src = tokenizer(
                src,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            self.examples.append(src["input_ids"])

            tgt = tokenizer(
                tgt,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
            self.targets.append(tgt["input_ids"])
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx], self.targets[idx]
