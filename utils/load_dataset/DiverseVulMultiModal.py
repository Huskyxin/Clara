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

        with open(f'/home/data/px/AI4SE/Mymodel/2025MutilModal/data/DiverseVul/Graph/DiverseVul-pdg-ALL.bin', 'rb') as f:
            data_all = pickle.load(f)


        data_list = [data_all[i] for i in self.n_idx]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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

def convert_examples_to_features(js, tokenizer, args, idx):
    
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

        
def load_indices(filename):
    with open(filename, 'r') as f:
        indices = [int(line.strip()) for line in f.readlines()]
    return indices

class DiverseVulDataset(Dataset):
    def __init__(self, tokenizer, args, mode='Train', file_path='/DiverseVul/function_div_modify.json'):
        self.graph_examples = []
        self.token_examples = []
        print('loading data...')

        if not os.path.exists(f'./data/{args.dataset}/{args.model}/'):
            os.makedirs(f'./data/{args.dataset}/{args.model}/')

        if not os.path.exists(f'./data/{args.dataset}/seed{args.dataseed}'):
            data_index_all = np.arange(330492)
            train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
            train_idxs, temp_index = train_test_split(data_index_all, test_size=(1 - train_ratio), random_state=args.dataseed)
            val_idxs, test_idxs = train_test_split(temp_index, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=args.dataseed)
            if not os.path.exists(f'./data/{args.dataset}/seed{args.dataseed}'):
                os.makedirs(f'./data/{args.dataset}/seed{args.dataseed}')
            np.savetxt(f'./data/{args.dataset}/seed{args.dataseed}/train_index.txt', train_idxs, fmt='%d')
            np.savetxt(f'./data/{args.dataset}/seed{args.dataseed}/val_index.txt', val_idxs, fmt='%d')
            np.savetxt(f'./data/{args.dataset}/seed{args.dataseed}/test_index.txt', test_idxs, fmt='%d')   
            train_idxs = train_idxs.tolist()
            val_idxs = val_idxs.tolist()
            test_idxs = test_idxs.tolist()
        else:
            train_idxs = load_indices(f'./data/{args.dataset}/seed{args.dataseed}/train_index.txt')
            val_idxs = load_indices(f'./data/{args.dataset}/seed{args.dataseed}/val_index.txt')
            test_idxs = load_indices(f'./data/{args.dataset}/seed{args.dataseed}/test_index.txt')
        
        if mode =='Train':
            file_name = f'{args.dataset}_seed{args.dataseed}_train.bin'
            binary_file_path = f'./data/{args.dataset}/{args.model}/'+file_name
            if os.path.exists(binary_file_path):
                with open(binary_file_path, 'rb') as file:
                    self.token_examples = pickle.load(file)
            else:
                with open(file_path) as f:
                    data = json.load(f)
                    for idx, js in tqdm(enumerate(data)):
                        self.token_examples.append(convert_examples_to_features(js, tokenizer, args, idx))
                self.token_example_tmp_train = [self.token_examples[i] for i in train_idxs]
                self.token_example_tmp_val = [self.token_examples[i] for i in val_idxs]
                self.token_example_tmp_test = [self.token_examples[i] for i in test_idxs]
                
                with open(binary_file_path, 'wb') as file:
                    pickle.dump(self.token_example_tmp_train, file)

                with open(f'./data/{args.dataset}/{args.model}/{args.dataset}_seed{args.dataseed}_val.bin', 'wb') as file:
                    pickle.dump(self.token_example_tmp_val, file)

                with open(f'./data/{args.dataset}/{args.model}/{args.dataset}_seed{args.dataseed}_test.bin', 'wb') as file:
                    pickle.dump(self.token_example_tmp_test, file)

                self.token_examples = [self.token_examples[i] for i in train_idxs]

        if mode =='Val':
            file_name = f'{args.dataset}_seed{args.dataseed}_val.bin'
            binary_file_path = f'./data/{args.dataset}/{args.model}/'+file_name
            if os.path.exists(binary_file_path):
                with open(binary_file_path, 'rb') as file:
                    self.token_examples = pickle.load(file)
            else:
                raise ValueError('No val data')
        if mode =='Test':
            file_name = f'{args.dataset}_seed{args.dataseed}_test.bin'
            binary_file_path = f'./data/{args.dataset}/{args.model}/'+file_name
            if os.path.exists(binary_file_path):
                with open(binary_file_path, 'rb') as file:
                    self.token_examples = pickle.load(file)
            else:
                raise ValueError('No test data')  
       
        tmp_path = f'./data/{args.dataset}/Graph/seed{args.dataseed}/{mode}/'
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        if mode == 'Train':
            pyg_data = VLGraphDataset(tmp_path, mode='Train', n_idx = train_idxs)
            self.graph_example = list(pyg_data)
        if mode =='Val':
            pyg_data = VLGraphDataset(tmp_path, mode='Val', n_idx = val_idxs)
            self.graph_example = list(pyg_data)
        if mode =='Test':
            pyg_data = VLGraphDataset(tmp_path, mode='Test', n_idx = test_idxs)
            self.graph_example = list(pyg_data)
            
    def __len__(self):
        return len(self.graph_example)
    def __getitem__(self, i):       
        return torch.tensor(self.token_examples[i].input_ids), torch.tensor(self.token_examples[i].label), self.graph_example[i]

