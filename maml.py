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
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from args import get_train_test_args

from tqdm import tqdm

import numpy as np
from train import *

#TODO: use a logger, use tensorboard
class MAML():
    def __init__(self, args, log):
        self.lr = args.lr
        self.outer_lr = args.outer_lr
        self.batch_size = args.batch_size
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

    def task(self, data):
        '''
        step 3 of MAML
        '''
        args = get_train_test_args()

        n = len(data)
        idx = list(range(n))
        np.random.shuffle(idx)

        # K-shot = support_p * data
        support_split_idx = int(np.floor(args.k_way))
        surport_idx, query_idx = idx[:support_split_idx], idx[support_split_idx:]

        support_sampler = SubsetRandomSampler(surport_idx)
        query_sampler = SubsetRandomSampler(query_idx)
        
        support_loader = DataLoader(data,
                                    batch_size=args.batch_size,
                                    sampler=support_sampler)
        query_loader = DataLoader(data,
                                  batch_size=args.batch_size,
                                  sampler=query_sampler)

        # print('========='+str(len(support_loader))+'=========')
        return support_loader, query_loader, surport_idx, query_idx

    # def get_dataset(self, args, datasets, data_dir, tokenizer, split_name):
    #     datasets = datasets.split(',')
    #     dataset_dict = None
    #     dataset_name=''
    #     for dataset in datasets:
    #         dataset_name += f'_{dataset}'
    #         dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
    #         dataset_dict = util.merge(dataset_dict, dataset_dict_curr)

    #     if args.device == torch.device('cpu'):
    #         print("Training locally")
    #         dataset_dict = self.task(args, dataset_dict)

    #     data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    #     return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict
    
    def inner_loop_train(self, inner_model, train_dataloader, eval_dataloader, val_dict):
        '''
        step 5 to 7
        '''

        device = self.device
        inner_model.to(device)
        inner_optim = AdamW(inner_model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        inner_loop_loss = 0
        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    inner_optim.zero_grad()
                    inner_model.train()
                    # print(batch)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = inner_model(input_ids, attention_mask=attention_mask,
                                          start_positions=start_positions,
                                          end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    inner_optim.step()
                    inner_optim.zero_grad()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx) 
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(inner_model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(inner_model)
                    global_idx += 1
                    inner_loop_loss += loss.item() 
        
        self.log.info(f"Inner Loop Average Loss: {inner_loop_loss/len(train_dataloader)}")
        return best_scores

    def inner_loop_q_sample(self, inner_model, query_batch):
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
        query_loss = query_outputs[0]
        query_loss.backward()
        inner_model.to(device)

        return query_outputs, query_loss, inner_model
                


def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    optim = AdamW(model.parameters(), lr=args.outer_lr)

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        maml = MAML(args, log)
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')

        sum_gradient = list()
        tasks_acc = list()
        for i in range(args.n_task):
            learner =  deepcopy(model)
 
            train_support_loader, train_query_loader, train_surport_idx, train_query_idx = maml.task(train_dataset)
            val_support_loader, val_query_loader, val_surport_idx, val_query_idx = maml.task(val_dataset)
            # val_dict_filtered = dict()
            # for key in val_dict.keys():
            #     print(key)
            #     print(val_surport_idx)
            #     print(val_dict[key][i][val_surport_idx[0]])
            #     val_dict_filtered[key] = [val_dict[key][i] for i in val_surport_idx]
            # val_dict_filtered = dict()
            # for key, value in val_dict.items():
            #     print(key)
            #     print(val_surport_idx)
            #     print(value[0])
            #     val_dict_filtered[key] = [value[idx] for idx in val_surport_idx]
            # print(len(val_dataset), len(val_dict_filtered['question']))


            best_scores = maml.inner_loop_train(learner, train_support_loader, val_support_loader, val_dict)

            for query_batch in train_query_loader:
                query_outputs, query_loss, learner = maml.inner_loop_q_sample(learner, query_batch)
                
                for i, params in enumerate(learner.parameters()):
                    if i == 0:
                        sum_gradient.append(deepcopy(params.grad))
                    else:
                        sum_gradient[i] += deepcopy(params.grad)
        
        query_logits = F.softmax(query_outputs[1], dim = 1)
        pre_label_id = torch.argmax(query_logits, dim = 1)
        pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
        q_label_id = q_label_id.detach().cpu().numpy().tolist()
        
        acc = accuracy_score(pre_label_id, q_label_id)
        tasks_acc.append(acc)
        for i in range(len(sum_gradient)):
            sum_gradient[i] /= args.n_task

        for i, params in enumerate(model.parameters()):
            params.grad = sum_gradient[i]

        optim.step()
        optim.zero_grad()
        maml.save(model)
        torch.cuda.empty_cache()

        
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
