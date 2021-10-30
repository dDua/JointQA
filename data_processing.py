import torch
import json
import os
import copy
import numpy as np
import random
from utils import get_data_loaders, process_all_contexts
from itertools import combinations

class HotpotQADataBase(object):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>"]

    def __init__(self, logger, args, tokenizer):
        self.args = args
        self.logger = logger
        self.tokenizer = tokenizer
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "question_offset",
                             "attention_mask", "token_type_ids", "answer_mask", "question_ids", "question_mask"]
        self.special_token_ids = tokenizer.convert_tokens_to_ids(self.special_tokens)
        if 'reasoning_file' in args and os.path.isfile(self.args.reasoning_file):
            self.reasoning_ann = json.load(open(self.args.reasoning_file))


    def get_instance(self, instance):
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2)
                                            - int(self.args.max_question_length))
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, pad_to_max_length=True,
                                                    max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]
        answer_output = np.array(answer_output)
        answer_output[answer_output == 0] = -100
        answer_output = answer_output.tolist()

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        sequence = ci_tokenized + cj_tokenized
        para_offset = len(sequence)
        sequence += question_tokens

        return {
            "input_ids": sequence,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offset  # accounted for bos tag in build_segments
        }

    def build_segments(self, data_point):
        token_types = [self.special_token_ids[0], data_point["input_ids"][0]]
        prev_token_type = token_types[-1]
        for input_i in data_point["input_ids"][1:]:
            if input_i in self.special_token_ids:
                prev_token_type = input_i
            token_types.append(prev_token_type)

        data_point["token_type_ids"] = token_types
        data_point["input_ids"] = [self.special_token_ids[0]] + data_point["input_ids"]
        data_point["attention_mask"] = [1]*len(token_types)
        data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_and_tensorize_dataset(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(x) for x in instances["input_ids"]))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))

        for name in self.model_inputs:
            max_n = max_q if "question" in name else max_l

            if "question_offset" in name or "answer" in name:
                continue
            for instance_name in instances[name]:
                instance_name += [padding] * (max_n - len(instance_name))

        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(instances[name]))

        return tensors

    def get_data_loaders(self, train=True, lazy=False, use_cache=True):
        return get_data_loaders(self, include_train=train, lazy=lazy, use_cache=use_cache)

    def get_reasoning_label(self, inst_id):
        rtype = self.reasoning_ann[inst_id] if inst_id in self.reasoning_ann else 2
        if rtype == 0:
            rlabel = "<comparison>"
        elif rtype == 1:
            rlabel = "<filter>"
        elif rtype == 3:
            rlabel = "<intersection>"
        else:
            rlabel = "<bridge>"

        return rtype, rlabel

class HotpotQADataAllPairs(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>", "<answer>",
                       "<reasoning>", "<filter>", "<bridge>", "<comparison>", "<intersection>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "question_offset", "attention_mask",
                             "token_type_ids", "answer_mask", "question_ids", "question_mask", "cij_labels"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts(self.tokenizer, instance, int(self.args.max_context_length/2 -
                                                                     int(self.args.max_question_length)),
                                            lowercase=self.args.lowercase)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]
        answer_output = np.array(answer_output)
        answer_output[answer_output == 0] = -100
        answer_output = answer_output.tolist()

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        para_offsets, cij_labels = [], []

        rtype, rtype_token = self.get_reasoning_label(instance["_id"])
        reasoning_toks = [self.tokenizer.convert_tokens_to_ids(rtype_token)]

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequence = ci_tokenized + cj_tokenized
        para_offsets.append(len(pos_sequence))
        pos_sequence += reasoning_toks
        pos_sequences = [[self.special_token_ids[0]] + pos_sequence + [self.special_token_ids[1]]]
        cij_labels.append(1)

        negative_rtypes = set(["<filter>", "<bridge>", "<comparison>", "<intersection>"]).difference(set([rtype_token]))
        negative_reasoning_toks = [self.tokenizer.convert_tokens_to_ids(neg_rt) for neg_rt in negative_rtypes]

        neg_sequences = []
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        random.shuffle(neg_pair_indices)

        if instance['mode'] == 'train':
            for neg_rtoks in negative_reasoning_toks:
                neg_toks = [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + [neg_rtoks, self.special_token_ids[1]]
                neg_sequences.append(neg_toks)
                para_offsets.append(len(neg_toks))
                cij_labels.append(1)

        for ci_idx, cj_idx in neg_pair_indices[:self.args.num_negative-len(neg_sequences)]:
            if [ci_idx, cj_idx] == sf_indices:
                continue
            else:
                ck_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"] + \
                               context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]
                neg_sequences.append([self.special_token_ids[0]]+ck_tokenized+reasoning_toks + [self.special_token_ids[1]])
                para_offsets.append(len(neg_sequences[-1]))
                cij_labels.append(0)

        all_input_ids = pos_sequences + neg_sequences

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offsets,  # accounted for bos tag in build_segments,
            "cij_labels": cij_labels
        }

    def build_segments(self, data_point):
        token_type_ids = []
        for sequence in data_point["input_ids"]:
            token_types = [self.special_token_ids[0], sequence[0]]
            prev_token_type = token_types[-1]
            for input_i in sequence[1:]:
                if input_i in self.special_token_ids:
                    prev_token_type = input_i
                token_types.append(prev_token_type)
            token_type_ids.append(token_types)

        data_point["token_type_ids"] = token_type_ids
        data_point["attention_mask"] = [[1]*len(token_types) for token_types in token_type_ids]
        data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        max_a = min(self.args.max_output_length + 1, max(len(x)+1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                continue
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-100] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif name == "question_offset":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-1] * (max_ns - len(instance_name))
                    instances[name][k] = instance_name[:max_ns]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]
        return instances

    def pad_instance_lazy(self, instances):
        padding = 0
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name == "cij_labels":
                padded_instances[name] = copy.deepcopy(instances[name])
                padded_instances[name] += [-1]*(max_ns - len(instances[name]))
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif "answer" in name:
                padded_instances[name] = (instances[name] + [-100] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = (instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
            else:
                padded_instances[name] = copy.deepcopy(instances[name])
                for k, sequence in enumerate(padded_instances[name]):
                    sequence += [padding] * (max_n - len(sequence))
                    padded_instances[name][k] = sequence[:max_n]
                padded_instances[name] += [[padding] * max_n] * (max_ns - len(padded_instances[name]))
                padded_instances[name] = padded_instances[name][:max_ns]
        return padded_instances

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors
