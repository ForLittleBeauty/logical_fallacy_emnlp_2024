
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
max_source_length = 511
max_target_length = 256
no_decay = ['bias', 'layer_norm.weight', 'LayerNorm.weight']
weight_decay = 1e-2
valid_steps = 512
batch_size = 1
num_epochs = 10
gradient_accumulation_steps = 4
warmup_proportion = 0
roberta_lr = 1e-5
tree_lr = 2e-5
t5_lr = 3e-5






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





''' logical relation connectives '''

conjuction = ['and', 'as well as', 'as well', 'also', 'separately']
alternative = ['or', 'either', 'instead', 'alternatively', 'else', 'nor', 'neither']
restatement = ['specifically', 'particularly', 'in particular', 'besides', 'additionally', 'in addition', 'moreover',
               'furthermore', 'further', 'plus', 'not only', 'indeed', 'in other words', 'in fact', 'in short',
               'in the end', 'overall', 'in sum', 'in summary', 'in detail', 'in details']
instantiation = ['for example', 'for instance', 'such as', 'including', 'as an example', 'for one thing']

contrast = ['but', 'however', 'yet', 'while', 'unlike', 'rather', 'rather than', 'in comparison', 'by comparison',
            'on the other hand', 'on the contrary', 'contrary to', 'in contrast', 'by contrast', 'still', 'whereas',
            'conversely', 'not', 'no', 'none', 'nothing', 'n\'t']
concession = ['although', 'though', 'despite', 'despite of', 'in spite of', 'regardless', 'regardless of', 'whether',
              'nevertheless', 'nonetheless', 'even if', 'even though', 'even as', 'even when', 'even after',
              'even so', 'no matter']
analogy = ['likewise', 'similarly', 'as if', 'as though', 'just as', 'just like', 'namely']

temporal = ['during', 'before', 'after', 'when', 'as soon as', 'then', 'next', 'until', 'till', 'meanwhile', 'in turn',
            'meantime', 'afterwards', 'afterward', 'simultaneously', 'at the same time', 'beforehand', 'previous',
            'previously', 'earlier', 'later', 'thereafter', 'finally', 'ultimately', 'eventually', 'subsequently']

condition = ['if', 'as long as', 'unless', 'otherwise', 'except', 'whenever', 'whichever', 'provided', 'once',
             'only if', 'only when', 'depend on', 'depends on', 'depending on', 'in case']
causal = ['because', 'cause', 'as a result', 'result in', 'due to', 'therefore', 'hence', 'thus', 'thereby', 'since',
          'now that', 'consequently', 'in consequence', 'in order to', 'so as to', 'so that', 'so', 'as', 'why', 'for',
          'accordingly', 'given', 'turn out', 'turns out']

all_keywords = conjuction + alternative + restatement + instantiation + contrast + concession + analogy + temporal + condition + causal




''' construct logical structure tree '''

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')


def recover_text_from_tree(constituency_tree):
    ''' extract the text from the given constituency tree '''

    text = " ".join(constituency_tree.leaves())

    return text


def match_keyword(constituency_tree, previous_matched_keyword_list):
    ''' from top-down and left-right traverse the constituency tree, find the leftmost longest matched keyword '''

    matched_flag = 0
    matched_keyword = ""
    matched_subtree = None

    for subtree in constituency_tree.subtrees():
        text_of_subtree = recover_text_from_tree(subtree)
        if text_of_subtree in all_keywords:
            presented_flag = 0
            for previous_matched_keyword in previous_matched_keyword_list:
                if text_of_subtree in previous_matched_keyword: # the matched keyword is part of or equal to the previous matched keyword
                    presented_flag = 1
                    break

            if presented_flag == 0:
                matched_flag = 1
                matched_keyword = text_of_subtree
                matched_subtree = subtree
                break

    return matched_flag, matched_keyword, matched_subtree


def extract_arguments(matched_keyword, matched_subtree):
    ''' extract the left and right argument of the matched keyword '''

    parent_text = recover_text_from_tree(matched_subtree.parent())

    right_argument = parent_text.split(matched_keyword)[-1]

    if parent_text.split(matched_keyword)[0] != "": # parent = alpha + keyword + betta
        left_argument = parent_text.split(matched_keyword)[0]
    else: # parent = keyword + betta
        if matched_subtree.parent().parent() is not None:
            grandparent_text = recover_text_from_tree(matched_subtree.parent().parent())
            if grandparent_text.split(parent_text)[0] == "":
                if matched_subtree.parent().parent().label()[:5] == "sent_":
                    if matched_subtree.parent().parent().left_sibling() is not None: # exist previous sentence
                        left_argument = recover_text_from_tree(matched_subtree.parent().parent().left_sibling())
                    else:
                        left_argument = ""
                else:
                    left_argument = ""
            else:
                left_argument = grandparent_text.split(parent_text)[0]
        else:
            left_argument = ""

    if len(right_argument) != 0: # delete starting and ending empty space
        if right_argument[0] == " ":
            right_argument = right_argument[1:]
        if right_argument[-1] == " ":
            right_argument = right_argument[:-1]

    if len(left_argument) != 0: # delete starting and ending empty space
        if left_argument[0] == " ":
            left_argument = left_argument[1:]
        if left_argument[-1] == " ":
            left_argument = left_argument[:-1]

    return left_argument, right_argument


def construct_logical_structure_tree(text):
    ''' construct the logical structure tree for the given text '''

    doc = nlp(text.lower())
    constituency_tree_string = "(Root"

    for sent_i in range(len(doc.sentences)):
        sentence_tree = doc.sentences[sent_i].constituency
        sentence_tree.label = "sent_" + str(sent_i)
        constituency_tree_string += " " + str(sentence_tree)

    constituency_tree_string += ")"
    constituency_tree = ParentedTree.fromstring(constituency_tree_string)


    logical_structure_tree = []
    # the logical relations in each sentence are saved from macro to micro perspective

    for sent_i in range(len(constituency_tree)):
        sentence_text = recover_text_from_tree(constituency_tree[sent_i])
        logical_structure_this_sentence = {"sentence_id": sent_i, "sentence_text": sentence_text, "logical_relation": []}

        # initialize
        previous_matched_keyword_list = []
        matched_flag = 1

        # recursively match keyword from the sub constituency_tree that corresponds to the betta argument
        while matched_flag != 0:
            matched_flag, matched_keyword, matched_subtree = match_keyword(constituency_tree[sent_i], previous_matched_keyword_list)
            if matched_flag == 1:
                previous_matched_keyword_list.append(matched_keyword)
                left_argument, right_argument = extract_arguments(matched_keyword, matched_subtree)
                logical_structure_this_sentence["logical_relation"].append({"logical_keyword": matched_keyword, "left_argument": left_argument, "right_argument": right_argument})

        logical_structure_tree.append(logical_structure_this_sentence)

    return logical_structure_tree





''' custom dataset '''

t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
tree_tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")


class custom_dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):

        text = self.dataframe['text'][idx]
        fallacy_label = self.dataframe['fallacy_label'][idx]

        logical_structure_tree = construct_logical_structure_tree(text)


        ''' pre_prompt, fallacy_list, fallacy_def '''

        instruction_prompt = "<s>The task is to detect whether the Text contains logical fallacy or not. "
        instruction_prompt += "The logical fallacy can be: " + ", ".join(fallacy_def) + ".\n"


        ''' textualized tree '''

        textualized_tree = "The logical relations in the Text are presented in this table: argument 1\tlogical relation\targument 2\n"

        for sent_i in range(len(logical_structure_tree)):
            for logic_relation_i in range(len(logical_structure_tree[sent_i]["logical_relation"]) - 1, -1, -1): # bottom-up textualize the tree
                logical_keyword = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i]["logical_keyword"]
                left_argument = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i]["left_argument"]
                right_argument = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i]["right_argument"]
                textualized_tree += left_argument + "\t" + logical_keyword + "\t" + right_argument + "\n"

        ''' source_input_ids, t5_soft_token '''

        source_input_ids = t5_tokenizer(instruction_prompt, add_special_tokens=False).input_ids
        # source_input_ids += t5_tokenizer(textualized_tree, add_special_tokens=False).input_ids
        source_input_ids += t5_tokenizer("Please answer \"Yes\" if the Text contains logical fallacy, else answer \"No\". Text: ", add_special_tokens=False).input_ids
        t5_soft_token = [len(source_input_ids)] # the index in source_input_ids where to add soft token
        source_input_ids += t5_tokenizer(text, add_special_tokens=False).input_ids
        source_input_ids += t5_tokenizer(" Answer:", add_special_tokens=False).input_ids if text[-1] == "." or text[-1] == "?" or text[-1] == "!" or text[-1] == "\n" else t5_tokenizer(". Answer:", add_special_tokens=False).input_ids

        if len(source_input_ids) > max_source_length: # truncate the text
            num_delete_token = len(source_input_ids) - max_source_length
            source_input_ids = t5_tokenizer(instruction_prompt, add_special_tokens=False).input_ids
            # source_input_ids += t5_tokenizer(textualized_tree, add_special_tokens=False).input_ids
            source_input_ids += t5_tokenizer("Please answer \"Yes\" if the Text contains logical fallacy, else answer \"No\". Text: ", add_special_tokens=False).input_ids
            t5_soft_token = [len(source_input_ids)]
            source_input_ids += t5_tokenizer(text, add_special_tokens=False).input_ids[:-num_delete_token]
            source_input_ids += t5_tokenizer(" Answer:", add_special_tokens=False).input_ids if text[-1] == "." or text[-1] == "?" or text[-1] == "!" or text[-1] == "\n" else t5_tokenizer(". Answer:", add_special_tokens=False).input_ids


        source_input_ids = torch.tensor(source_input_ids)
        source_input_ids = source_input_ids.view(1, source_input_ids.shape[0])
        t5_soft_token = torch.tensor(t5_soft_token)


        ''' target_input_ids '''

        target_input_ids = t5_tokenizer(fallacy_label + "</s>", add_special_tokens=False).input_ids
        target_input_ids = torch.tensor(target_input_ids)
        target_input_ids = target_input_ids.view(1, target_input_ids.shape[0])


        ''' tree_input_ids, tree_attention_mask '''

        # used to derive logical structure tree embedding

        logical_keyword_list = []
        left_argument_list = []
        right_argument_list = []
        sentence_text_list = []

        num_logical_relations_each_sent = [] # the number of logical relations in each sentence
        index_logical_relation = [] # the index of logical relation that the logical keyword belongs to

        for sent_i in range(len(logical_structure_tree)):
            num_logical_relations_each_sent.append(len(logical_structure_tree[sent_i]["logical_relation"]))
            sentence_text_list.append(logical_structure_tree[sent_i]["sentence_text"])

            for logic_relation_i in range(len(logical_structure_tree[sent_i]["logical_relation"]) - 1, -1, -1):
                logical_keyword = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i]["logical_keyword"]
                left_argument = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i]["left_argument"]
                right_argument = logical_structure_tree[sent_i]["logical_relation"][logic_relation_i]["right_argument"]

                logical_keyword_list.append(" " + logical_keyword) # add space before, for roberta tokenizer
                left_argument_list.append(" " + left_argument)
                right_argument_list.append(" " + right_argument)

                if logical_keyword in conjuction:
                    index_logical_relation.append(1)
                if logical_keyword in alternative:
                    index_logical_relation.append(2)
                if logical_keyword in restatement:
                    index_logical_relation.append(3)
                if logical_keyword in instantiation:
                    index_logical_relation.append(4)
                if logical_keyword in contrast:
                    index_logical_relation.append(5)
                if logical_keyword in concession:
                    index_logical_relation.append(6)
                if logical_keyword in analogy:
                    index_logical_relation.append(7)
                if logical_keyword in temporal:
                    index_logical_relation.append(8)
                if logical_keyword in condition:
                    index_logical_relation.append(9)
                if logical_keyword in causal:
                    index_logical_relation.append(10)

        if len(sentence_text_list) != 0:
            sentence_text_input_ids = tree_tokenizer(sentence_text_list, add_special_tokens=False, padding='longest')["input_ids"]
            sentence_text_attention_mask = tree_tokenizer(sentence_text_list, add_special_tokens=False, padding='longest')["attention_mask"]

            sentence_text_input_ids = torch.tensor(sentence_text_input_ids)
            sentence_text_attention_mask = torch.tensor(sentence_text_attention_mask)

            num_logical_relations_each_sent = torch.tensor(num_logical_relations_each_sent)
        else:
            sentence_text_input_ids = torch.tensor([])
            sentence_text_attention_mask = torch.tensor([])
            num_logical_relations_each_sent = torch.tensor([])


        if len(logical_keyword_list) != 0:
            logical_keyword_input_ids = tree_tokenizer(logical_keyword_list, add_special_tokens=False, padding='longest')["input_ids"]
            logical_keyword_attention_mask = tree_tokenizer(logical_keyword_list, add_special_tokens=False, padding='longest')["attention_mask"]
            left_argument_input_ids = tree_tokenizer(left_argument_list, add_special_tokens=False, padding='longest')["input_ids"]
            left_argument_attention_mask = tree_tokenizer(left_argument_list, add_special_tokens=False, padding='longest')["attention_mask"]
            right_argument_input_ids = tree_tokenizer(right_argument_list, add_special_tokens=False, padding='longest')["input_ids"]
            right_argument_attention_mask = tree_tokenizer(right_argument_list, add_special_tokens=False, padding='longest')["attention_mask"]

            logical_keyword_input_ids = torch.tensor(logical_keyword_input_ids)
            logical_keyword_attention_mask = torch.tensor(logical_keyword_attention_mask)
            left_argument_input_ids = torch.tensor(left_argument_input_ids)
            left_argument_attention_mask = torch.tensor(left_argument_attention_mask)
            right_argument_input_ids = torch.tensor(right_argument_input_ids)
            right_argument_attention_mask = torch.tensor(right_argument_attention_mask)

            index_logical_relation = torch.tensor(index_logical_relation)
        else:
            logical_keyword_input_ids = torch.tensor([])
            logical_keyword_attention_mask = torch.tensor([])
            left_argument_input_ids = torch.tensor([])
            left_argument_attention_mask = torch.tensor([])
            right_argument_input_ids = torch.tensor([])
            right_argument_attention_mask = torch.tensor([])
            index_logical_relation = torch.tensor([])



        dict = {"source_input_ids": source_input_ids, "target_input_ids": target_input_ids, "t5_soft_token": t5_soft_token,
                "logical_keyword_input_ids": logical_keyword_input_ids, "logical_keyword_attention_mask": logical_keyword_attention_mask,
                "left_argument_input_ids": left_argument_input_ids, "left_argument_attention_mask": left_argument_attention_mask,
                "right_argument_input_ids": right_argument_input_ids, "right_argument_attention_mask": right_argument_attention_mask,
                "sentence_text_input_ids": sentence_text_input_ids, "sentence_text_attention_mask": sentence_text_attention_mask,
                "num_logical_relations_each_sent": num_logical_relations_each_sent, "index_logical_relation": index_logical_relation}

        return dict





''' model '''

class Tree_Embedding(nn.Module):

    def __init__(self):
        super(Tree_Embedding, self).__init__()

        self.roberta = RobertaModel.from_pretrained("FacebookAI/roberta-base", output_hidden_states=True, )

        self.projection_layer_1 = nn.Linear(768, 1024, bias=True)
        nn.init.xavier_uniform_(self.projection_layer_1.weight)
        nn.init.zeros_(self.projection_layer_1.bias)

        self.projection_layer_2 = nn.Linear(1024, 1024, bias=True)
        nn.init.xavier_uniform_(self.projection_layer_2.weight)
        nn.init.zeros_(self.projection_layer_2.bias)

        self.W_text_tree = nn.Linear(768 * 2, 768, bias=True)
        nn.init.xavier_uniform_(self.W_text_tree.weight)
        nn.init.zeros_(self.W_text_tree.bias)

        self.W_conjuction = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_conjuction.weight)
        nn.init.zeros_(self.W_conjuction.bias)

        self.W_alternative = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_alternative.weight)
        nn.init.zeros_(self.W_alternative.bias)

        self.W_restatement = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_restatement.weight)
        nn.init.zeros_(self.W_restatement.bias)

        self.W_instantiation = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_instantiation.weight)
        nn.init.zeros_(self.W_instantiation.bias)

        self.W_contrast = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_contrast.weight)
        nn.init.zeros_(self.W_contrast.bias)

        self.W_concession = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_concession.weight)
        nn.init.zeros_(self.W_concession.bias)

        self.W_analogy = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_analogy.weight)
        nn.init.zeros_(self.W_analogy.bias)

        self.W_temporal = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_temporal.weight)
        nn.init.zeros_(self.W_temporal.bias)

        self.W_condition = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_condition.weight)
        nn.init.zeros_(self.W_condition.bias)

        self.W_causal = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_causal.weight)
        nn.init.zeros_(self.W_causal.bias)

    def token_embedding(self, roberta_input_ids, roberta_attention_mask): # input size: batch size * number of tokens

        outputs = self.roberta(input_ids=roberta_input_ids, attention_mask=roberta_attention_mask)
        hidden_states = outputs[2]
        token_embeddings_layers = torch.stack(hidden_states, dim=0)  # 13 layer * batch_size * number of tokens * 768
        token_embeddings = torch.sum(token_embeddings_layers[-4:, :, :, :], dim=0) # sum up the last four layers, batch_size * number of tokens * 768

        return token_embeddings

    def sequence_embedding(self, roberta_input_ids, roberta_attention_mask):

        token_embeddings_in_sequence = self.token_embedding(roberta_input_ids, roberta_attention_mask) # batch size * number of tokens * 768
        sum_token_embeddings = torch.sum(torch.mul(token_embeddings_in_sequence, roberta_attention_mask.view(roberta_attention_mask.shape[0], roberta_attention_mask.shape[1], 1).repeat(1, 1, 768)), dim=1)
        num_none_padding_tokens = torch.sum(roberta_attention_mask, dim=1).view(roberta_attention_mask.shape[0], 1).repeat(1, 768)
        meal_pooling_embedding = torch.div(sum_token_embeddings, num_none_padding_tokens) # batch size * 768

        return meal_pooling_embedding

    def subtree_embedding(self, left_argument, logical_keyword, right_argument, lower_subtree, lower_subtree_flag, logical_relation):

        if lower_subtree_flag == 1:
            right_argument_combine = self.W_text_tree(torch.cat((right_argument, lower_subtree), dim=1))
            concat_embedding = torch.cat((left_argument, logical_keyword, right_argument_combine), dim=1)
        else:
            concat_embedding = torch.cat((left_argument, logical_keyword, right_argument), dim=1)

        if logical_relation == 1:
            return self.W_conjuction(concat_embedding)
        if logical_relation == 2:
            return self.W_alternative(concat_embedding)
        if logical_relation == 3:
            return self.W_restatement(concat_embedding)
        if logical_relation == 4:
            return self.W_instantiation(concat_embedding)
        if logical_relation == 5:
            return self.W_contrast(concat_embedding)
        if logical_relation == 6:
            return self.W_concession(concat_embedding)
        if logical_relation == 7:
            return self.W_analogy(concat_embedding)
        if logical_relation == 8:
            return self.W_temporal(concat_embedding)
        if logical_relation == 9:
            return self.W_condition(concat_embedding)
        if logical_relation == 10:
            return self.W_causal(concat_embedding)

    def forward(self, logical_keyword_input_ids, logical_keyword_attention_mask, left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask, sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation):

        tree_embedding = torch.zeros((1, 768)).to(device)

        if sentence_text_input_ids.shape[0] != 0:
            sentence_text_embedding = self.sequence_embedding(sentence_text_input_ids, sentence_text_attention_mask)  # number of sentences * 768
            num_sents = sentence_text_embedding.shape[0]

            if logical_keyword_input_ids.shape[0] != 0:
                logical_keyword_embedding = self.sequence_embedding(logical_keyword_input_ids, logical_keyword_attention_mask)  # number of logical relations * 768
                left_argument_embedding = self.sequence_embedding(left_argument_input_ids, left_argument_attention_mask)
                right_argument_embedding = self.sequence_embedding(right_argument_input_ids, right_argument_attention_mask)

                for sent_i in range(num_sents):
                    num_logical_relations_this_sent = num_logical_relations_each_sent[sent_i].item()

                    start_index_of_logical_relation = 0 if sent_i == 0 else torch.sum(num_logical_relations_each_sent[:sent_i]).item()
                    end_index_of_logical_relation = torch.sum(num_logical_relations_each_sent[:(sent_i + 1)]).item()

                    logical_keyword_embedding_this_sent = logical_keyword_embedding[start_index_of_logical_relation:end_index_of_logical_relation, :]
                    left_argument_embedding_this_sent = left_argument_embedding[start_index_of_logical_relation:end_index_of_logical_relation, :]
                    right_argument_embedding_this_sent = right_argument_embedding[start_index_of_logical_relation:end_index_of_logical_relation, :]
                    index_logical_relation_this_sent = index_logical_relation[start_index_of_logical_relation:end_index_of_logical_relation]

                    if num_logical_relations_this_sent != 0:
                        sentence_tree_embedding = self.subtree_embedding(left_argument_embedding_this_sent[0, :].view(1, 768), logical_keyword_embedding_this_sent[0, :].view(1, 768), right_argument_embedding_this_sent[0, :].view(1, 768),
                                                                         torch.zeros((1, 768)).to(device), 0, index_logical_relation_this_sent[0].item())

                        for logic_relation_i in range(1, num_logical_relations_this_sent): # num_logical_relations_this_sent == logical_keyword_embedding_this_sent.shape[0]
                            sentence_tree_embedding = self.subtree_embedding(left_argument_embedding_this_sent[logic_relation_i, :].view(1, 768), logical_keyword_embedding_this_sent[logic_relation_i, :].view(1, 768), right_argument_embedding_this_sent[logic_relation_i, :].view(1, 768),
                                                                             sentence_tree_embedding, 1, index_logical_relation_this_sent[logic_relation_i].item())

                        tree_embedding += self.W_text_tree(torch.cat((sentence_text_embedding[sent_i, :].view(1, 768), sentence_tree_embedding), dim=1))

                    else:
                        tree_embedding += sentence_text_embedding[sent_i, :].view(1, 768)

            else:
                for sent_i in range(num_sents):
                    tree_embedding += sentence_text_embedding[sent_i, :].view(1, 768)

            tree_embedding = tree_embedding / num_sents

        tree_embedding = self.projection_layer_2(self.projection_layer_1(tree_embedding))

        return tree_embedding



class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        self.tree = Tree_Embedding()

    def forward(self, source_input_ids, target_input_ids, t5_soft_token, logical_keyword_input_ids, logical_keyword_attention_mask, left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask, sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation, inference_mode):

        tree_embedding = self.tree(logical_keyword_input_ids, logical_keyword_attention_mask, left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask, sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation)
        tree_embedding = tree_embedding.view(1, tree_embedding.shape[0], tree_embedding.shape[1])

        source_input_embeds = self.t5.shared(source_input_ids)

        ''' add tree embedding as soft token '''

        new_source_input_embeds = source_input_embeds[:, :t5_soft_token.item(), :]
        new_source_input_embeds = torch.cat((new_source_input_embeds, tree_embedding), dim=1)
        new_source_input_embeds = torch.cat((new_source_input_embeds, source_input_embeds[:, t5_soft_token.item():, :]), dim=1)

        if inference_mode == 1:
            outputs = self.t5.generate(inputs_embeds=new_source_input_embeds, max_new_tokens=max_target_length)
            return outputs
        else:
            loss = self.t5(inputs_embeds=new_source_input_embeds, labels=target_input_ids).loss
            return loss






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
        t5_soft_token = batch["t5_soft_token"][0]
        logical_keyword_input_ids = batch["logical_keyword_input_ids"][0]
        logical_keyword_attention_mask = batch["logical_keyword_attention_mask"][0]
        left_argument_input_ids = batch["left_argument_input_ids"][0]
        left_argument_attention_mask = batch["left_argument_attention_mask"][0]
        right_argument_input_ids = batch["right_argument_input_ids"][0]
        right_argument_attention_mask = batch["right_argument_attention_mask"][0]
        sentence_text_input_ids = batch["sentence_text_input_ids"][0]
        sentence_text_attention_mask = batch["sentence_text_attention_mask"][0]
        num_logical_relations_each_sent = batch["num_logical_relations_each_sent"][0]
        index_logical_relation = batch["index_logical_relation"][0]

        source_input_ids, target_input_ids, t5_soft_token, logical_keyword_input_ids, logical_keyword_attention_mask, left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask, sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation = \
            source_input_ids.to(device), target_input_ids.to(device), t5_soft_token.to(device), logical_keyword_input_ids.to(device), logical_keyword_attention_mask.to(device), left_argument_input_ids.to(device), left_argument_attention_mask.to(device), right_argument_input_ids.to(device), right_argument_attention_mask.to(device), sentence_text_input_ids.to(device), sentence_text_attention_mask.to(device), num_logical_relations_each_sent.to(device), index_logical_relation.to(device)

        label_text = t5_tokenizer.decode(target_input_ids[0])

        try:
            # inference
            with torch.no_grad():
                outputs = model(source_input_ids, target_input_ids, t5_soft_token, logical_keyword_input_ids, logical_keyword_attention_mask,
                                left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask,
                                sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation, inference_mode=1)

            generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
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


param_all = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('roberta' in n))], 'lr': roberta_lr, 'weight_decay': weight_decay},
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('tree' in n) and (not 'roberta' in n))], 'lr': tree_lr, 'weight_decay': weight_decay},
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('t5' in n))], 'lr': t5_lr, 'weight_decay': weight_decay},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('roberta' in n))], 'lr': roberta_lr, 'weight_decay': 0.0},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('tree' in n) and (not 'roberta' in n))], 'lr': tree_lr, 'weight_decay': 0.0},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('t5' in n))], 'lr': t5_lr, 'weight_decay': 0.0}]
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
                torch.save(model.state_dict(), "./saved_models/flan_t5_identify_structure.ckpt")
                best_fallacy_F_dev = fallacy_F


        model.train()

        source_input_ids = batch["source_input_ids"][0]
        target_input_ids = batch["target_input_ids"][0]
        t5_soft_token = batch["t5_soft_token"][0]
        logical_keyword_input_ids = batch["logical_keyword_input_ids"][0]
        logical_keyword_attention_mask = batch["logical_keyword_attention_mask"][0]
        left_argument_input_ids = batch["left_argument_input_ids"][0]
        left_argument_attention_mask = batch["left_argument_attention_mask"][0]
        right_argument_input_ids = batch["right_argument_input_ids"][0]
        right_argument_attention_mask = batch["right_argument_attention_mask"][0]
        sentence_text_input_ids = batch["sentence_text_input_ids"][0]
        sentence_text_attention_mask = batch["sentence_text_attention_mask"][0]
        num_logical_relations_each_sent = batch["num_logical_relations_each_sent"][0]
        index_logical_relation = batch["index_logical_relation"][0]

        with accelerator.accumulate(model):

            loss = model(source_input_ids, target_input_ids, t5_soft_token, logical_keyword_input_ids, logical_keyword_attention_mask,
                         left_argument_input_ids, left_argument_attention_mask, right_argument_input_ids, right_argument_attention_mask,
                         sentence_text_input_ids, sentence_text_attention_mask, num_logical_relations_each_sent, index_logical_relation, inference_mode=0)

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
        torch.save(model.state_dict(), "./saved_models/flan_t5_identify_structure.ckpt")
        best_fallacy_F_dev = fallacy_F



# test

model.load_state_dict(torch.load("./saved_models/flan_t5_identify_structure.ckpt", map_location=device))
fallacy_precision, fallacy_recall, fallacy_F, micro_F = evaluate(model, test_dataloader, verbose=1)

















# stop here
