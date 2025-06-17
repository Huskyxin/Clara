import os
import datetime
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator
import importlib
import multiprocessing
from tqdm import tqdm, trange
import json
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import math
import argparse
import numpy as np
import argparse
import logging
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from train.mm_learning_trainer_accelerate import *
import importlib
from utils.load_dataset.PrimeVulMultiModal import PrimeVulDataset
from utils.load_dataset.DiverseVulMultiModal import DiverseVulDataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
import warnings
warnings.filterwarnings("ignore")
from accelerate import load_checkpoint_in_model
logger = get_logger(__name__)
from utils.misc import *
from accelerate import DistributedDataParallelKwargs

def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset', type=str, default="DiverseVul",
                        help="using dataset from this project.")
    parser.add_argument('--model_dir', type=str, required=True,
                        help="directory to store the model weights.")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='save', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--test_cwe', type=str, default=None,
                        required=False, help="using dataset from this CWE for testing.")

    # Graph model parameters
    parser.add_argument("--num_node_features", default=100, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--node_hidden_dim", default=200, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    # Other parameters
    parser.add_argument("--max_source_length", default=400, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--discribe", default="prime", type=str,
                        help="The description of the model.")
    parser.add_argument("--model", default="codegen", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default='', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--model_path_local", default='', type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", type=str2bool, default="n",
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                        "The training dataset will be truncated in block of this size for training."
                        "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--load_huggface", type=str2bool, default="t",
                        help="Whether to load the pretrained model from huggingface")    
    parser.add_argument("--do_train", type=str2bool, default="n",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", type=str2bool, default="n",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", type=str2bool, default="n",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_best_test", type=str2bool, default="n",
                        help="Whether to run eval on the dev set.")  
    parser.add_argument("--joint_train", type=str2bool, default="n",
                        help="Whether to run eval on the dev set.")  
    parser.add_argument("--save_test", type=str2bool, default="n",
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--do_test_prob", type=str2bool, default="n",
                        help="Whether to run eval and save the prediciton probabilities.")
    parser.add_argument("--val_during_training", type=str2bool, default="n",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--test_during_training", type=str2bool, default="n",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--log_train", type=str2bool, default="y",
                        help="Whether to log")
    parser.add_argument("--save_best_metric", type=str, default="f1",
                        help="The metric to be used to save the best model.")    

    parser.add_argument("--do_lower_case", type=str2bool, default="n",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--weighted_sampler", type=str2bool, default="n",
                        help="Whether to do project balanced sampler using WeightedRandomSampler.")
    # Soft F1 loss function
    parser.add_argument("--soft_f1", type=str2bool, default="n",
                        help="Use soft f1 loss instead of regular cross entropy loss.")
    parser.add_argument("--class_weight", type=str2bool, default="n",
                        help="Use class weight in the regular cross entropy loss.")
    parser.add_argument("--vul_weight", default=1.0, type=float,
                        help="Weight for the vulnerable class in the regular cross entropy loss.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup ratio over all steps.")
    parser.add_argument("--vdscore_value", default=0.005, type=float,
                        help="漏洞检测分数的阈值")
    
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", type=str2bool, default="n",
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", type=str2bool, default="n",
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', type=str2bool, default="n",
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', type=str2bool, default="n",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--dataseed', type=int, default=9527,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--max-patience', type=int, default=-1,
                        help="Max iterations for model with no improvement.")

    # preprocess parameters
    parser.add_argument('--n_qp', default=0, type=int, help='length of query prompt vectors')
    # parser.add_argument('--n_qcp', default=8, type=int, help='length of query context prompt vectors')
    parser.add_argument('--n_context', default=8, type=int, help='length of query context prompt vectors')
    parser.add_argument('--n_fuse', default=0, type=int, help='length of fusion context prompt vectors')
    parser.add_argument('--n_fusion_layers', default=1, type=int, help='number of multimodal fusion layers')
    parser.add_argument('--loss_weight', default=2, type=int)
    
    # GT parameters
    parser.add_argument('--num_heads', type=int,
                        default=4, help="number of heads")
    parser.add_argument('--num_layers', type=int,
                        default=6, help="number of layers")
    parser.add_argument('--dim_hidden', type=int, default=768,
                        help="hidden dimension of Transformer")
    parser.add_argument('--gt_dropout', type=float, default=0.2, help="dropout")    

    # MModal parameters
    parser.add_argument("--pretrain_batch_size", default=16, type=int)
    parser.add_argument("--graph_pretrain_epoch", default=20, type=int)
    parser.add_argument('--graph_prelearning_rate', type=float, default=1e-4)
    parser.add_argument("--pregradient_accumulation_steps", default=4, type=int)
    
    parser.add_argument('--trans_learning_rate', type=float, default=2e-5)
    parser.add_argument('--graph_learning_rate', type=float, default=2e-5)
    parser.add_argument('--token_max_grad_norm', type=float, default=2.0)
    parser.add_argument('--graph_max_grad_norm', type=float, default=2.0)
    parser.add_argument('--graph_pre_max_grad_norm', type=float, default=10.0)
    parser.add_argument('--load_pretrain', type=str2bool, default="n")    

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    args.block_size = args.block_size - args.n_context - args.n_fuse
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],
                            gradient_accumulation_steps=args.gradient_accumulation_steps)

    device = accelerator.device
    args.n_gpu = accelerator.num_processes
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size 
    args.per_gpu_eval_batch_size=args.eval_batch_size // args.n_gpu
    # Setup logging
    if not os.path.exists(f'./save/logs/{args.model_dir}'):
        os.makedirs(f'./save/logs/{args.model_dir}')
    current_date = datetime.datetime.now()
    current_date = current_date.replace(microsecond=0)
    logging.basicConfig(filename=f'./save/logs/{args.model_dir}/{args.dataset}_{current_date}.log',
                        filemode="w",
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        encoding='utf-8')
    logger.info(accelerator.state, main_process_only=False)    
    
    set_random_seed(args.seed)
    set_seed(args.seed)
    args.start_epoch = 0
    args.start_step = 0    
    
    # Setup Model
    config_class, model_class, tokenizer_class = AutoConfig, AutoModel, AutoTokenizer
    if args.load_huggface == True:
        config = config_class.from_pretrained(args.model_name_or_path,
                                            cache_dir=args.cache_dir)
        if args.model not in ["codegen"]:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                    do_lower_case=args.do_lower_case, # 区分大小写字母，
                                                        cache_dir=args.cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                    cache_dir=args.cache_dir,
                                                    trust_remote_code=True)
            
        if args.model in ["starcoder"]:
            llm_model = model_class.from_pretrained(args.model_name_or_path,
                                                cache_dir=args.cache_dir,
                                                torch_dtype = torch.bfloat16,
                                                attn_implementation = "flash_attention_2")
            
        elif args.model in ["codegen"]:
            llm_model = model_class.from_pretrained(args.model_name_or_path,
                                                cache_dir=args.cache_dir,
                                                torch_dtype = torch.bfloat16,)
        else:
            llm_model = model_class.from_pretrained(args.model_name_or_path,
                                                cache_dir=args.cache_dir,)
                                                
    else:
        config = config_class.from_pretrained(args.model_path_local)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path_local)
        llm_model = model_class.from_pretrained(args.model_path_local)
        config.pad_token_id = tokenizer(tokenizer.pad_token, 
                                        truncation=True)['input_ids'][0]
    
    module_name = args.model
    module = importlib.import_module('models.'+module_name)
    model_class = getattr(module, 'Model')

    model = model_class(llm_model, config, tokenizer, args)

    logger.info("*"*40, main_process_only=False)

    max_len = 0
    for key, value in args.__dict__.items(): 
        if max_len < len(key):
            max_len = len(key)
    for key, value in args.__dict__.items(): 
        logger.info(key+(max_len-len(key))*' '+": "+str(value), main_process_only=False) 

    logger.info("*"*40, main_process_only=False)

    if args.do_train:
        with accelerator.main_process_first():
            if args.dataset == 'PrimeVul':
                test_dataset = PrimeVulDataset(tokenizer, args, 'Test', args.test_data_file)
                eval_dataset = PrimeVulDataset(tokenizer, args, 'Val', args.eval_data_file)
                train_dataset = PrimeVulDataset(tokenizer, args, 'Train', args.train_data_file)
            elif args.dataset == 'DiverseVul':
                train_dataset = DiverseVulDataset(tokenizer, args, mode='Train')
                eval_dataset = DiverseVulDataset(tokenizer, args, mode='Val')
                test_dataset = DiverseVulDataset(tokenizer, args, mode='Test')
            else:
                raise ValueError("Invalid dataset")
        train(args, accelerator, train_dataset, eval_dataset, test_dataset, model, logger)
        
    if args.save_test:
        with accelerator.main_process_first():
            if args.dataset == 'PrimeVul':
                test_dataset = PrimeVulDataset(args, args.test_data_file)
                eval_dataset = PrimeVulDataset(args, args.eval_data_file)
            elif args.dataset == 'DiverseVul':
                test_dataset = DiverseVulDataset(tokenizer, args, mode='Test')
                eval_dataset = DiverseVulDataset(tokenizer, args, mode='Val')
        model = accelerator.prepare(model)
        model = accelerator.unwrap_model(model)

        path = './save/pytorch_model.bin'
        load_checkpoint_in_model(model, path)
        model = accelerator.prepare(model)
        save_test(args, accelerator, test_dataset, model, logger, mode = 'Test')


if __name__ == "__main__":
    main()

