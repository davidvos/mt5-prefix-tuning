import json
import os

from torch.utils.data import Dataset
import torch

class WebNLGDataset(Dataset):
    
    def __init__(self, tokenizer, raw_path='../webnlg_data/release_v3.0/ru', language='en', data_path='../webnlg_data/preprocessed', split='train'):

        self.tokenizer = tokenizer
        
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
            src = tokenizer.encode(src)
            if len(src)>512:
                # Truncate
                encoded = src[:511] + [tokenizer.eos_token_id]
            self.examples.append(src)
            tgt = tokenizer.encode(tgt)
            self.targets.append(tgt)

    def collate_fn(self, batch):
        """
        Same Sequence Length on Same Batch
        """
        max_len_data=0
        max_len_label=0
        for data, label in batch:
            if len(data)>max_len_data: max_len_data=len(data)
            if len(label)>max_len_label: max_len_label=len(label)
                
        datas=[]
        attn_masks=[]
        labels=[]
        for data, label in batch:
            data.extend([self.tokenizer.pad_token_id]*(max_len_data-len(data)))
            datas.append(data)
            
            attn_mask=[int(e!=self.tokenizer.pad_token_id) for e in data]
            attn_masks.append(attn_mask)
            
            label.extend([-100]*(max_len_label-len(label)))
            labels.append(label)
                    
        return torch.tensor(datas), torch.tensor(attn_masks), torch.tensor(labels)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx], self.targets[idx]
