import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter

from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler
from torch.nn import functional as F
from torch.autograd import Variable
from args import get_train_test_args

from tqdm import tqdm

import numpy as np
from train import *



def read_and_process_task(args, tokenizer, dataset_dict, dir_name, dataset_name, split, task):
    #TODO: cache this if possible
    # cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    # if os.path.exists(cache_path) and not args.recompute_features:
    #     tokenized_examples = util.load_pickle(cache_path)
    # else:
    if split=='train':
        tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
    else:
        tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
    # util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples

#TODO: use a logger, use tensorboard
class MAML():
    def __init__(self, args, log):
        self.lr = args.lr
        self.outer_lr = args.outer_lr
        self.batch_size = args.batch_size
        self.ideal_batch_size = args.ideal_batch_size
        self.outer_batch_size = args.outer_batch_size
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions

        if not os.path.exists(self.path):
            os.makedirs(self.path)
    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def outer_evaluate(self, model, data_loader_list, data_dict_list, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        total_length = 0
        for data_loader in data_loader_list:
            total_length += len(data_loader.dataset)

        with torch.no_grad(), \
                tqdm(total=total_length) as progress_bar:
            for data_loader, data_dict in zip(data_loader_list, data_dict_list):
                for batch in data_loader:
                    # Setup for forward
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    batch_size = len(input_ids)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    # Forward
                    start_logits, end_logits = outputs.start_logits, outputs.end_logits
                    # TODO: compute loss

                    all_start_logits.append(start_logits)
                    all_end_logits.append(end_logits)
                    progress_bar.update(batch_size)

                # Get F1 and EM scores
                start_logits = torch.cat(all_start_logits).cpu().numpy()
                end_logits = torch.cat(all_end_logits).cpu().numpy()
                preds = util.postprocess_qa_predictions(data_dict,
                                                        data_loader.dataset.encodings,
                                                        (start_logits, end_logits))


        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def task(self, data_dict):
        '''
        step 3 of MAML
        '''

        args = get_train_test_args()

        n = len(data_dict['id'])
        idx = list(range(n))
        np.random.seed()
        np.random.shuffle(idx)
        # print(idx[0:5])

        # K-shot 
        print(f"{args.k_shot}-shot learning")
        support_split_idx = int(args.k_shot)
        support_idx, query_idx = idx[:support_split_idx], idx[support_split_idx:support_split_idx*2]

        support_dict = dict()
        query_dict = dict()
        for key in data_dict.keys():
            support_dict[key] = [data_dict[key][i] for i in support_idx]
            query_dict[key] = [data_dict[key][i] for i in query_idx]

        return support_idx, query_idx, support_dict, query_dict

    def get_task_dataset(self, args, datasets, data_dir, tokenizer, split_name, task):
        '''
        step 3 of MAML
        '''

        datasets = datasets.split(',')
        dataset_dict = None
        dataset_name=''
        for dataset in datasets:
            dataset_name += f'_{dataset}'
            dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
            dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
        

        support_idx, query_idx, support_dict, query_dict = self.task(dataset_dict)
        # print(f'======= support idx: {support_idx}, query idx: {query_idx} =======')

        support_data_encodings = read_and_process_task(args, tokenizer, support_dict, data_dir, dataset_name, split_name, task)
        query_data_encodings = read_and_process_task(args, tokenizer, query_dict, data_dir, dataset_name, split_name, task)
        return util.QADataset(support_data_encodings, train=(split_name=='train')), support_dict, support_idx, util.QADataset(query_data_encodings, train=(split_name=='train')), query_dict, query_idx
    
    def inner_loop_train(self, inner_model, train_dataloader, val_dataloader, val_dict, global_idx, best_scores):
        '''
        step 5 to 7
        '''

        device = self.device
        inner_model.to(device)
        inner_optim = AdamW(inner_model.parameters(), lr=self.lr)

        inner_loop_loss = 0
        self.log.info(f'============ Inner Train starts ============')
        with torch.enable_grad():
            step = 0
            for batch in train_dataloader:
                # inner_optim.zero_grad()
                inner_model.train()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = inner_model(input_ids, attention_mask=attention_mask,
                                        start_positions=start_positions,
                                        end_positions=end_positions)
                
                loss = outputs[0]
                loss.backward()
                if (step % self.ideal_batch_size) == 0:
                    inner_optim.step()
                    inner_optim.zero_grad()

                step += self.batch_size
                global_idx += 1
                inner_loop_loss += loss.item() 
        
        self.log.info(f"Inner Loop Average Loss: {inner_loop_loss/len(train_dataloader)}")
        return best_scores, global_idx

    def inner_loop_query(self, inner_model, query_batch):
        '''
        step 8
        '''
        device = self.device
        query_input_ids = query_batch['input_ids'].to(device)
        query_attention_mask = query_batch['attention_mask'].to(device)
        query_start_positions = query_batch['start_positions'].to(device)
        query_end_positions = query_batch['end_positions'].to(device)
        query_outputs = inner_model(query_input_ids, attention_mask=query_attention_mask,
                                    start_positions=query_start_positions,
                                    end_positions=query_end_positions)

        return query_outputs, inner_model

    def inner_loop(self, args, model, tokenizer, optim, val_loader, val_dict, global_idx, best_scores, progress_bar):
        '''
        step 4 to 9
        '''
        model.to(args.device)
        sum_gradient = list()

        query_list = list()
        query_dict = list()
        for task_id in range(args.n_task):
            learner =  deepcopy(model)
            progress_bar.update(args.k_shot)

            if args.oodomain == True:
                if (task_id % int(1/args.p_oodomain_task)) == 0:
                    self.log.info(f"Preparing Training Data for Task {task_id} from oodomain...")
                    train_support_dataset, train_support_dict, train_support_idx, train_query_dataset, train_query_dict, train_query_idx = self.get_task_dataset(args, args.eval_datasets, args.oodomain_train_dir, tokenizer, 'train', task_id)
                else:
                    self.log.info(f"Preparing Training Data for Task {task_id} ...")
                    train_support_dataset, train_support_dict, train_support_idx, train_query_dataset, train_query_dict, train_query_idx = self.get_task_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train', task_id)
                
            else:
                self.log.info(f"Preparing Training Data for Task {task_id} ...")
                train_support_dataset, train_support_dict, train_support_idx, train_query_dataset, train_query_dict, train_query_idx = self.get_task_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train', task_id)
                
            # print(f'=====task: {task_id}, support idx: {train_support_idx}, query idx: {train_query_idx}')
            train_support_loader = DataLoader(train_support_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_support_dataset))

            train_query_loader = DataLoader(train_query_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(train_query_dataset)) 

 
            best_scores, global_idx = self.inner_loop_train(learner, train_support_loader, val_loader, val_dict, global_idx, best_scores)
            

            sum_query_loss = 0
            for query_batch in train_query_loader:
                query_outputs, learner = self.inner_loop_query(learner, query_batch)
                query_loss = Variable(query_outputs[0], requires_grad = True)
                query_loss.backward()
                learner.to(args.device)
                sum_query_loss += query_loss.item() 
            self.log.info(f"Query Average Loss: {sum_query_loss/len(train_query_loader)}")

            num_params = 0
            for i, params in enumerate(learner.parameters()):
                if task_id == 0:
                    sum_gradient.append(deepcopy(params.grad))
                else:
                    sum_gradient[i] += deepcopy(params.grad) 
                num_params += 1

            del learner
            torch.cuda.empty_cache()


            query_list.append(train_query_loader)
            query_dict.append(train_query_dict)
            
            print('===='*20)

        for i in range(len(sum_gradient)):
            sum_gradient[i] /= args.n_task

        for i, params in enumerate(model.parameters()):
            params.grad = sum_gradient[i]

        return query_loss, train_query_loader, train_query_idx, global_idx, query_list, query_dict
            




def main():
    # define parser and arguments
    args = get_train_test_args()

    # util.set_seed(args.seed)
    # model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    # model = DistilBertForQuestionAnswering.from_pretrained("/Users/chi-hsuanchang/Documents/cs224n/robustqa/save_vm/save/mod_baseline-01/checkpoint")
    model = DistilBertForQuestionAnswering.from_pretrained("/home/achchg/robustqa/save/mod_baseline-01/checkpoint")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    optim = AdamW(model.parameters(), lr=args.outer_lr)


    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


        log.info(f"Preparing Validation Data...")
        tbx = SummaryWriter(args.save_dir)
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
                                
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))

        maml = MAML(args, log)
        
        # outer_loop (meta_step)
        outer_global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}

        for epoch_num in range(args.outer_num_epochs):
            log.info(f'Meta Epoch: {epoch_num}')
            log.info('='*20)
            with tqdm(total=args.k_shot*args.n_task) as progress_bar:
                query_loss, train_query_loader, train_query_idx, global_idx, query_list, query_dict = maml.inner_loop(args, model, tokenizer, optim, val_loader, val_dict, outer_global_idx, best_scores, progress_bar)
                outer_global_idx += global_idx/args.n_task

                progress_bar.set_postfix(epoch=epoch_num, NLL=query_loss.item())
                tbx.add_scalar('train/NLL', query_loss.item(), global_idx) 

                if args.indomain_eval == False:
                    log.info(f'Evaluating at step {global_idx}... on query set')
                    preds, curr_score = maml.outer_evaluate(model, query_list, query_dict, return_preds=True)

                    agg_query_dict = dict()
                    for d in query_dict: 
                        for key, value in d.items():
                            try:
                                agg_query_dict[key] += value
                            except:
                                agg_query_dict[key] = value


                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                    log.info('Visualizing in TensorBoard...')
                    for k, v in curr_score.items():
                        tbx.add_scalar(f'query val/{k}', v, global_idx)
                    log.info(f'Eval - query {results_str}')
                    if args.visualize_predictions:
                        util.visualize(tbx,
                                    pred_dict=preds,
                                    gold_dict=agg_query_dict,
                                    step=global_idx,
                                    split='val',
                                    num_visuals=args.num_visuals)
                    if curr_score['F1'] >= best_scores['F1']:
                        best_scores = curr_score
                        maml.save(model)

                    preds_val, curr_score_val = maml.evaluate(model, val_loader, val_dict, return_preds=True)
                    results_str_val = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score_val.items())
                    log.info('Visualizing in TensorBoard...')
                    for k, v in curr_score_val.items():
                        tbx.add_scalar(f'val/{k}', v, global_idx)
                    log.info(f'Eval {results_str_val}')
                    if args.visualize_predictions:
                        util.visualize(tbx,
                                    pred_dict=preds_val,
                                    gold_dict=val_dict,
                                    step=global_idx,
                                    split='val',
                                    num_visuals=args.num_visuals)

                else:
                    log.info(f'Evaluating at step {global_idx}...')
                    preds_val, curr_score_val = maml.evaluate(model, val_loader, val_dict, return_preds=True)

                    results_str_val = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score_val.items())
                    log.info('Visualizing in TensorBoard...')
                    for k, v in curr_score_val.items():
                        tbx.add_scalar(f'val/{k}', v, global_idx)
                    log.info(f'Eval {results_str_val}')
                    if args.visualize_predictions:
                        util.visualize(tbx,
                                    pred_dict=preds_val,
                                    gold_dict=val_dict,
                                    step=global_idx,
                                    split='val',
                                    num_visuals=args.num_visuals)
                    if curr_score_val['F1'] >= best_scores['F1']:
                        best_scores = curr_score_val
                        maml.save(model)
                
                optim.step()
                optim.zero_grad()
                maml.save(model)

        
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
