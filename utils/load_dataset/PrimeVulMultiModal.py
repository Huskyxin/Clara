import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import torch
import json
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from torch_geometric.data import InMemoryDataset

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label

def convert_examples_to_features(js, tokenizer, args):
    
    if args.model == 'svudl' or 'unixcoder':
        code = js['func']
        code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
        source_tokens = [tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length
    else:
        code = js['func']
        code_tokens = tokenizer.tokenize(code)
        if '</s>' in code_tokens:
            code_tokens = code_tokens[:code_tokens.index('</s>')]
        source_tokens = code_tokens[:args.block_size]
        source_ids = tokenizer.encode(js['func'].split("</s>")[0], max_length=args.block_size, padding='max_length', truncation=True)
    return InputFeatures(source_tokens, source_ids, js['idx'], js['target'])


class VLGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, mode='Train', n_idx = None):
        self.mode = mode
        self.n_idx = n_idx
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):

        return []

    @property
    def processed_file_names(self):
        return [f'graph.pt']

    def download(self):
        pass

    def process(self):

        with open(f'/data/DiverseVul/Graph/DiverseVul-pdg-ALL.bin', 'rb') as f:
            data_all = pickle.load(f)


        data_list = [data_all[i] for i in self.n_idx]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
def load_indices(filename):
    with open(filename, 'r') as f:
        indices = [int(line.strip()) for line in f.readlines()]
    return indices

class PrimeVulDataset(Dataset):
    def __init__(self, tokenizer, args, mode='Train', file_path='/DiverseVul/function_div_modify.json'):
        self.graph_examples = []

        print('loading data...')

        file_name_with_extension = os.path.basename(file_path)


        file_name = os.path.splitext(file_name_with_extension)[0]
        file_name = file_name+'.bin'
        binary_file_path = f'./data/PrimeVul/{args.model}/'+file_name

        if not os.path.exists(f'./data/PrimeVul/{args.model}/'):
            os.makedirs(f'./data/PrimeVul/{args.model}/')

        self.token_example = []
        if os.path.exists(binary_file_path):
            with open(binary_file_path, 'rb') as file:
                self.token_example = pickle.load(file)
        else:
            with open(file_path) as f:
                for line in f:
                    js=json.loads(line.strip())
                    self.token_example.append(convert_examples_to_features(js, tokenizer, args))
            with open(binary_file_path, 'wb') as file:
                pickle.dump(self.token_example, file)

        if mode =='Train':
            file_name = f'{args.dataset}-pdg-clip-train.bin'
        elif mode =='Val':
            file_name = f'{args.dataset}-pdg-clip-val.bin'
        elif mode =='Test':
            file_name = f'{args.dataset}-pdg-clip-test.bin'
        binary_file_path = f'./data/{args.dataset}/Graph/'+file_name
        with open(binary_file_path, 'rb') as file:
            self.graph_example = pickle.load(file)

    def __len__(self):
        return len(self.graph_example)
    def __getitem__(self, i):       
        return torch.tensor(self.token_example[i].input_ids), torch.tensor(self.token_example[i].label), self.graph_example[i]

