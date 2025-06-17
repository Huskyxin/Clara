
import copy
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import load_checkpoint_in_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from torch_geometric.data import Batch


def data_collate_fn(batch):
    """Collate examples into batch for training."""
    data1 = [item[0] for item in batch]
    data2 = [item[1] for item in batch]
    data3 = [item[2] for item in batch]

    data1 = torch.stack(data1, dim=0)
    data2 = torch.stack(data2, dim=0)
    batch_data3 = Batch.from_data_list(data3)

    return data1, data2, batch_data3

def calculate_metrics(labels, preds):
    acc=accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    TN, FP, FN, TP = confusion_matrix(labels, preds).ravel()
    tnr = TN/(TN+FP)
    fpr = FP/(FP+TN)
    fnr = FN/(TP+FN)
    return round(acc,4)*100, round(prec,4)*100, \
        round(recall,4)*100, round(f1,4)*100, round(tnr,4)*100, \
            round(fpr,4)*100, round(fnr,4)*100

def test(args, accelerator, eval_dataset, model, logger, mode = 'Test'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_dataset = TextDataset(args, args.test_data_file)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size, collate_fn=data_collate_fn)
    
    eval_dataloader = accelerator.prepare(eval_dataloader)

    logger.info("***** Load Best Model *****")
    logger.info(f"***** Running {mode} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    logits=[]   
    labels=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        # x, edge_index, edge_attr, batch, label = batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y
        label = batch.y
        if args.vul_weight != 1.0:
            weight = torch.tensor([1.0, args.vul_weight]).to(accelerator.device)
        else:
            weight = None
        with torch.no_grad():
            lm_loss, logit = model(batch, label, weight)  
        logit, label = accelerator.gather_for_metrics((logit, label))
        logits.append(logit.cpu().float().numpy())
        labels.append(label.cpu().float().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,1]>0.5

    test_acc, test_prec, test_recall, test_f1, test_tnr, test_fpr, test_fnr = calculate_metrics(labels, preds)

    accelerator.wait_for_everyone()

    result = {
        f"{mode}_acc": test_acc,
        f"{mode}_prec": test_prec,
        f"{mode}_recall": test_recall,
        f"{mode}_f1": test_f1,
        f"{mode}_tnr": test_tnr,
        f"{mode}_tpr": test_fpr,
        f"{mode}_fnr": test_fnr,
    }
    for key, value in result.items():
        logger.info("Best_%s:  %s = %s", mode, key, round(value, 4))                     
    return result 

def save_test(args, accelerator, eval_dataset, model, logger, mode = 'Test'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_dataset = TextDataset(args, args.test_data_file)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size, collate_fn=data_collate_fn)
    
    eval_dataloader = accelerator.prepare(eval_dataloader)

    logger.info("***** Load Best Model *****")
    logger.info(f"***** Running {mode} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    logits=[]   
    labels=[]
    for batch in eval_dataloader:
        inputs, label, graph = batch
        # label = batch.y
        if args.vul_weight != 1.0:
            weight = torch.tensor([1.0, args.vul_weight]).to(accelerator.device)
        else:
            weight = None
        with torch.no_grad():
            lm_loss, logit = model(inputs, graph, label, weight)
        logit, label = accelerator.gather_for_metrics((logit, label))
        logits.append(logit.cpu().float().numpy())
        labels.append(label.cpu().float().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,1]>0.5
    vuln_scores = logits[:,1].tolist()

    def load_indices(filename):
        with open(filename, 'r') as f:
            indices = [int(line.strip()) for line in f.readlines()]
        return indices
    idxs = load_indices(f'./data/val_index.txt')
    
    import pandas as pd
    results_df = pd.DataFrame({
        'Idx': idxs,
        'True_Label': labels.astype(int),
        'Predicted_Label': preds.astype(int),
    })

    # 预测结果保存位置
    checkpoint_prefix = f'Best_Pred_idxs/{args.dataset}'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))      
    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(output_dir + f'/{args.model_dir}+{args.discribe}.csv', index=False)
    
    test_acc, test_prec, test_recall, test_f1, test_tnr, test_fpr, test_fnr = calculate_metrics(labels, preds)
    accelerator.wait_for_everyone()

    result = {
        f"{mode}_acc": test_acc,
        f"{mode}_prec": test_prec,
        f"{mode}_recall": test_recall,
        f"{mode}_f1": test_f1,
        f"{mode}_tnr": test_tnr,
        f"{mode}_tpr": test_fpr,
        f"{mode}_fnr": test_fnr
        # f"{mode}_VD_S": vds,
        # "vdscore_value": args.vdscore_value,
        # "VD_S_Threshold": threshold
    }
    for key, value in result.items():
        logger.info("Best_%s:  %s = %s", mode, key, round(value, 4))                    
        
    return result 

def train(args, accelerator, train_dataset, eval_dataset, test_dataset, model, logger):
    logger.info("  ***** training mode *****")
    """ Train the model """

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=4, collate_fn=data_collate_fn)
    pretrain_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.pretrain_batch_size, num_workers=4, collate_fn=data_collate_fn)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=4, collate_fn=data_collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=4, collate_fn=data_collate_fn)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_update_steps_per_epoch_ = math.ceil(len(pretrain_dataloader) / args.pregradient_accumulation_steps)
    args.max_steps = args.epoch*num_update_steps_per_epoch
    args.max_steps_graphpre = args.graph_pretrain_epoch*num_update_steps_per_epoch_
    args.save_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch

    optimizer = AdamW(model.parameters(), lr=args.trans_learning_rate, eps=args.adam_epsilon)

    if args.warmup_steps == 0:
        num_warmup = args.max_steps * args.warmup_ratio
    else:
        num_warmup = args.warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup,
                                                num_training_steps=args.max_steps)


    train_dataloader, eval_dataloader, test_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader,eval_dataloader, test_dataloader, model, optimizer, scheduler
    )
    model.zero_grad()

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    
    logger.info("  ***** Running *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                total_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_token_f1= 0.0
    test_token_best_f1 = 0.0
    best_acc=0.0
    patience = 0

    step = 1
    model.zero_grad()
    logger.info("  ***** Running training *****")
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader, total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        tr_num = 0
        train_loss = 0
        logits_lst_t = []
        labels_lst_t = []

        for local_step, batch in enumerate(bar):
            inputs, labels, graph = batch
            model.train()
            with accelerator.accumulate(model):
                if args.vul_weight != 1.0:
                    weight = torch.tensor([1.0, args.vul_weight]).to(accelerator.device)
                else:
                    weight = None
                # train graph level
                loss, logit = model(inputs, graph, labels, weight)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()  
            
            logit_t, labels = accelerator.gather_for_metrics((logit, labels))

            # cast to torch.float16
            logits_lst_t.append(logit_t.detach().cpu().float().numpy())
            labels_lst_t.append(labels.detach().cpu().float().numpy())

            step_t_loss = accelerator.reduce(loss.detach().clone()).item()

            step_loss = step_t_loss
            tr_loss += step_loss
            tr_num += 1
            train_loss += step_loss
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            
            ###
            # log after every logging_steps (e.g., 1000)
            ###
            if (step) % args.logging_steps == 0:
                avg_loss=round(train_loss/tr_num, 5)
                # train_acc, train_prec, train_recall, train_f1, train_tnr, train_fpr, train_fnr = calculate_metrics(step_labels_lst, step_preds_lst)
                if args.val_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    
                    token_results = evaluate(args, accelerator, eval_dataloader, eval_dataset, model, logger, eval_when_training=True)
                    for key, value in token_results.items():
                        logger.info("Joint  epoch %s, Val_Step-%s:  %s = %s", idx, step, key, round(value, 4)) 

                    checkpoint_prefix = f'checkpoint-best-f1/{args.dataset}+{args.model_dir}+{args.discribe}/Joint'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    accelerator.wait_for_everyone()
                    accelerator.save_model(model, output_dir, safe_serialization=False)
                    logger.info(f"Saving valid f1 model checkpoint at epoch {idx} step {step} to {output_dir}")
                    accelerator.wait_for_everyone()

                    if token_results['eval_f1']>best_token_f1:
                        best_token_f1 = token_results['eval_f1']
                        logger.info("  "+"*"*20)
                        logger.info("  Joint  Best f1:%s",round(best_token_f1,4))
                        logger.info("  "+"*"*20)
            step += 1
            torch.cuda.empty_cache()
            

    logger.info("  "+"*"*20)
    logger.info("  Best Joint Test F1:%s",round(test_token_best_f1,4))
    logger.info("  "+"*"*20)

from sklearn.metrics import roc_auc_score
def evaluate(args, accelerator, eval_dataloader, eval_dataset, model, logger, eval_when_training=False):
    
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    losses = []
    logits = [] 
    labels = []
    for batch in eval_dataloader:
        inputs, label, graph = batch
        with torch.no_grad():
            if args.vul_weight != 1.0:
                weight = torch.tensor([1.0, args.vul_weight]).to(accelerator.device)
            else:
                weight = None

            lm_loss, logit = model(inputs, graph, label, weight)
        
        losses.append(accelerator.gather_for_metrics(lm_loss.repeat(args.eval_batch_size)))
        logit, label = accelerator.gather_for_metrics((logit, label))
        logits.append(logit.cpu().float().numpy())
        labels.append(label.cpu().float().numpy())
    
    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,1]>0.5
    vuln_scores = logits[:,1]

    eval_acc, eval_prec, eval_recall, eval_f1, eval_tnr, eval_fpr, eval_fnr = calculate_metrics(labels, preds)

    with accelerator.main_process_first():
        ground_truth = labels
        pred = vuln_scores
        vds, threshold = calculate_vul_det_score(pred, ground_truth, target_fpr=args.vdscore_value)
        auc = roc_auc_score(ground_truth, pred)
    perplexity = eval_loss.clone().detach()
    
    result = {
        "eval_loss": float(perplexity),
        "eval_acc": eval_acc,
        "eval_prec": eval_prec,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
        "vds": vds,
        "threshold": threshold,
        "auc": auc,
    }
    return result

from sklearn.metrics import roc_curve
def calculate_vul_det_score(predictions, ground_truth, target_fpr=0.005):
    """
    Calculate the vulnerability detection score (VD-S) given a tolerable FPR.
    
    Args:
    - predictions: List of model prediction probabilities for the positive class.
    - ground_truth: List of ground truth labels, where 1 means vulnerable class, and 0 means benign class.
    - target_fpr: The tolerable false positive rate.
    
    Returns:
    - vds: Calculated vulnerability detection score given the acceptable .
    - threshold: The classification threashold for vulnerable prediction.
    """
    
    # Calculate FPR, TPR, and thresholds using ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
    
    # Filter thresholds where FPR is less than or equal to the target FPR
    valid_indices = np.where(fpr <= target_fpr)[0]
    
    # Choose the threshold with the largest FPR that is still below the target FPR, if possible
    if len(valid_indices) > 0:
        idx = valid_indices[-1]  # Last index where FPR is below or equal to target FPR
    else:
        # If no such threshold exists (unlikely), default to the closest to the target FPR
        idx = np.abs(fpr - target_fpr).argmin()
        
    chosen_threshold = thresholds[idx]
    
    # Classify predictions based on the chosen threshold
    classified_preds = [1 if pred >= chosen_threshold else 0 for pred in predictions]
    
    # Calculate VD-S
    fn = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 0])
    tp = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 1])
    vds = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return vds, chosen_threshold