
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'


import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import RobertaTokenizer, RobertaModel
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel, PeftModelForSeq2SeqLM
import math
import random
import time
import datetime
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup
from scipy.optimize import linear_sum_assignment
from math import floor
from accelerate import Accelerator
import stanza
from nltk.tree import Tree, ParentedTree




''' hyper-parameters '''

dataset_name = "argotario"
max_source_length = 1024
max_target_length = 256
no_decay = ['bias', 'layernorm.weight']
weight_decay = 1e-2
valid_steps = 512
batch_size = 1
num_epochs = 10
gradient_accumulation_steps = 4
warmup_proportion = 0
llama_lr = 3e-4
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05







''' fallacy list and definition '''

if dataset_name == "argotario":
    fallacy_list = ['Ad Hominem', 'Emotional Language', 'Hasty Generalization', 'Irrelevant Authority', 'Red Herring']
    fallacy_def = ['Ad Hominem (the text attack a person instead of arguing against the claims)',
                   'Emotional Language (the text arouse non-rational emotions)',
                   'Hasty Generalization (the text draw a broad conclusion based on a limited sample of population)',
                   'Irrelevant Authority (the text cite an authority but the authority lacks relevant expertise)',
                   'Red Herring (the text diverge the attention to irrelevant issues)']

if dataset_name == "reddit":
    fallacy_list = ['Slippery Slope', 'Irrelevant Authority', 'Hasty Generalization', 'Black-and-White Fallacy',
                    'Ad Populum', 'Tradition Fallacy', 'Naturalistic Fallacy', 'Worse Problem Fallacy']
    fallacy_def = ['Slippery Slope (the text suggest taking a small initial step leads to a chain of related events culminating in significant effect)',
                   'Irrelevant Authority (the text cite an authority but the authority lacks relevant expertise)',
                   'Hasty Generalization (the text draw a broad conclusion based on a limited sample of population)',
                   'Black-and-White Fallacy (the text present two alternative options as the only possibilities)',
                   'Ad Populum (the text affirm something is true because the majority thinks so)',
                   'Tradition Fallacy (the text argue the action has always been done in the tradition)',
                   'Naturalistic Fallacy (the text claim something is good or bad because it is natural or unnatural)',
                   'Worse Problem Fallacy (the text justify an issue by arguing more severe issues exists)']

if dataset_name == "climate":
    fallacy_list = ['Evading Burden of Proof', 'Cherry Picking', 'Red Herring', 'Strawman', 'Irrelevant Authority',
                    'Hasty Generalization', 'False Cause', 'False Analogy', 'Vagueness']
    fallacy_def = ['Evading Burden of Proof (the text make a claim without evidence or supporting argument)',
                   'Cherry Picking (the text selectively present partial evidence to support a claim)',
                   'Red Herring (the text diverge the attention to irrelevant issues)',
                   'Strawman (the text distort the claim to another one to make it easier to attack)',
                   'Irrelevant Authority (the text cite an authority but the authority lacks relevant expertise)',
                   'Hasty Generalization (the text draw a broad conclusion based on a limited sample of population)',
                   'False Cause (the text assume two correlated events must also have a causal relation)',
                   'False Analogy (the text assume two alike things must be alike in other aspects)',
                   'Vagueness (the text use ambiguous words, terms, or phrases)']






''' read data '''

def filter_data(split_set):
    original_data = pd.read_csv("./processed_datasets/" + dataset_name + "_" + split_set + ".tsv", sep='\t', header=0)
    text_list = []
    fallacy_label_list = []
    for row_i in range(original_data.shape[0]):
        if original_data['fallacy_label'][row_i] in fallacy_list:
            text_list.append(original_data['text'][row_i])
            fallacy_label_list.append("Yes")
        if original_data['fallacy_label'][row_i] == "No Fallacy":
            text_list.append(original_data['text'][row_i])
            fallacy_label_list.append("No")

    filtered_data = pd.DataFrame({"fallacy_label": fallacy_label_list, "text": text_list})
    return filtered_data


train = filter_data("train")
dev = filter_data("dev")
test = filter_data("test")







''' custom dataset '''

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tree_tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")


class custom_dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):

        text = self.dataframe['text'][idx]
        fallacy_label = self.dataframe['fallacy_label'][idx]


        ''' pre_prompt, fallacy_list, fallacy_def '''

        instruction_prompt = "<s>The task is to detect whether the Text contains logical fallacy or not. "
        instruction_prompt += "The logical fallacy can be: " + ", ".join(fallacy_def) + ".\n"


        ''' source_input_ids, llama_soft_token '''

        source_input_ids = llama_tokenizer(instruction_prompt, add_special_tokens=False).input_ids
        source_input_ids += llama_tokenizer("Please answer \"Yes\" if the Text contains logical fallacy, else answer \"No\". Text: ", add_special_tokens=False).input_ids
        source_input_ids += llama_tokenizer(text, add_special_tokens=False).input_ids
        source_input_ids += llama_tokenizer(" Answer:", add_special_tokens=False).input_ids if text[-1] == "." or text[-1] == "?" or text[-1] == "!" or text[-1] == "\n" else llama_tokenizer(". Answer:", add_special_tokens=False).input_ids

        if len(source_input_ids) > max_source_length:
            num_delete_token = len(source_input_ids) - max_source_length
            source_input_ids = llama_tokenizer(instruction_prompt, add_special_tokens=False).input_ids
            source_input_ids += llama_tokenizer("Please answer \"Yes\" if the Text contains logical fallacy, else answer \"No\". Text: ", add_special_tokens=False).input_ids
            source_input_ids += llama_tokenizer(text, add_special_tokens=False).input_ids[:-num_delete_token]
            source_input_ids += llama_tokenizer(" Answer:", add_special_tokens=False).input_ids if text[-1] == "." or text[-1] == "?" or text[-1] == "!" or text[-1] == "\n" else llama_tokenizer(". Answer:", add_special_tokens=False).input_ids

        source_input_ids = torch.tensor(source_input_ids)
        source_input_ids = source_input_ids.view(1, source_input_ids.shape[0])


        ''' target_input_ids '''

        target_input_ids = llama_tokenizer(fallacy_label + "</s>", add_special_tokens=False).input_ids
        target_input_ids = torch.tensor(target_input_ids)
        target_input_ids = target_input_ids.view(1, target_input_ids.shape[0])


        dict = {"source_input_ids": source_input_ids, "target_input_ids": target_input_ids}

        return dict





''' model '''

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        lora_config = LoraConfig(r=lora_rank, target_modules=['q_proj', 'v_proj'], lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
        self.llama = get_peft_model(self.llama, lora_config)

    def forward(self, source_input_ids, target_input_ids, inference_mode):

        sequence_input_ids = torch.cat((source_input_ids, target_input_ids), dim=1)
        label_input_ids = torch.cat((torch.full((source_input_ids.shape[0], source_input_ids.shape[1]), -100, dtype=torch.int64).to(device), target_input_ids), dim=1)

        if inference_mode == 1:
            outputs = self.llama.generate(input_ids=source_input_ids, max_new_tokens=max_target_length)
            return outputs
        else:
            loss = self.llama(input_ids=sequence_input_ids, labels=label_input_ids).loss
            return loss

    def print_trainable_params(self):

        trainable_params = 0
        all_params = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_params






''' evaluate '''

def convert_label(text):

    if "Yes" in text:
        return 1
    if "No" in text:
        return 0

    return -1


def evaluate(model, eval_dataloader, verbose):

    model.eval()

    true_label = []
    prediction = []

    step_error = []

    for step, batch in enumerate(eval_dataloader):

        source_input_ids = batch["source_input_ids"][0]
        target_input_ids = batch["target_input_ids"][0]

        source_input_ids, target_input_ids = source_input_ids.to(device), target_input_ids.to(device)

        label_text = llama_tokenizer.decode(target_input_ids[0])

        try:
            # inference
            with torch.no_grad():
                outputs = model(source_input_ids, target_input_ids, inference_mode=1)

            generated_text = llama_tokenizer.decode(outputs[0][source_input_ids.shape[1]:], skip_special_tokens=True) # remove tokens in source prompt
            predicted_label = convert_label(generated_text)
            if predicted_label == -1:
                step_error.append(step)
            else:
                prediction.append(predicted_label)
                true_label.append(convert_label(label_text))

        except:
            step_error.append(step)


    if len(step_error) != 0:
        print("step error is ", len(step_error))

    if len(prediction) != 0:

        fallacy_precision = precision_recall_fscore_support(true_label, prediction, average='binary')[0]
        fallacy_recall = precision_recall_fscore_support(true_label, prediction, average='binary')[1]
        fallacy_F = precision_recall_fscore_support(true_label, prediction, average='binary')[2]
        micro_F = precision_recall_fscore_support(true_label, prediction, average='micro')[2]

        if verbose:
            print("Fallacy: ", precision_recall_fscore_support(true_label, prediction, average='binary'))
            print("Micro: ", precision_recall_fscore_support(true_label, prediction, average='micro'))

    else:
        fallacy_precision = 0
        fallacy_recall = 0
        fallacy_F = 0
        micro_F = 0

    return fallacy_precision, fallacy_recall, fallacy_F, micro_F










''' train '''

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()



seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.use_deterministic_algorithms(True, warn_only=True)


model = Model()
model.cuda()
trainable_params, all_params = model.print_trainable_params()
print("all_params is {:}, trainable_params is {:}, ratio of trainable_params is {:}".format(all_params, trainable_params, 100 * trainable_params / all_params))


param_all = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_all if (not any(nd in n for nd in no_decay))], 'lr': llama_lr, 'weight_decay': weight_decay},
    {'params': [p for n, p in param_all if (any(nd in n for nd in no_decay))], 'lr': llama_lr, 'weight_decay': 0.0}]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)


train_dataset = custom_dataset(train)
dev_dataset = custom_dataset(dev)
test_dataset = custom_dataset(test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


num_train_steps = num_epochs * len(train_dataloader) // gradient_accumulation_steps # scheduler.step_with_optimizer = True by default
warmup_steps = int(warmup_proportion * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)


accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)


best_fallacy_F_dev = 0

for epoch_i in range(num_epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0
    num_batch = 0  # number of batch to calculate average loss
    total_num_batch = 0  # number of batch in this epoch

    for batch in train_dataloader:

        if total_num_batch % valid_steps == 0 and total_num_batch != 0:

            # valid every valid_steps, actual update steps = valid_steps / gradient_accumulation_steps

            elapsed = format_time(time.time() - t0)
            avg_loss = total_loss / num_batch if num_batch != 0 else 0
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    loss average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_loss))

            total_loss = 0
            num_batch = 0

            fallacy_precision, fallacy_recall, fallacy_F, micro_F = evaluate(model, dev_dataloader, verbose=0)

            if fallacy_F > best_fallacy_F_dev:
                torch.save(model.state_dict(), "./saved_models/llama_identify_baseline.ckpt")
                best_fallacy_F_dev = fallacy_F



        model.train()

        source_input_ids = batch["source_input_ids"][0]
        target_input_ids = batch["target_input_ids"][0]

        with accelerator.accumulate(model):

            loss = model(source_input_ids, target_input_ids, inference_mode=0)

            total_loss += loss.item()
            num_batch += 1
            total_num_batch += 1

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()



    # valid at the end of each epoch

    elapsed = format_time(time.time() - t0)
    avg_loss = total_loss / num_batch if num_batch != 0 else 0
    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    loss average: {:.3f}'.format(total_num_batch, len(train_dataloader), elapsed, avg_loss))

    total_loss = 0
    num_batch = 0

    fallacy_precision, fallacy_recall, fallacy_F, micro_F = evaluate(model, dev_dataloader, verbose=0)

    if fallacy_F > best_fallacy_F_dev:
        torch.save(model.state_dict(), "./saved_models/llama_identify_baseline.ckpt")
        best_fallacy_F_dev = fallacy_F




# test

model.load_state_dict(torch.load("./saved_models/llama_identify_baseline.ckpt", map_location=device))
fallacy_precision, fallacy_recall, fallacy_F, micro_F = evaluate(model, test_dataloader, verbose=1)











# stop here
