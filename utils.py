import os
import json
import torch
import tqdm
import random
import traceback
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset

def process_all_contexts(tokenizer, instance, max_passage_len, sf_only=False, add_sent_ends=False, lowercase=True):
    context_info = []
    sf_titles = {}

    for sf in instance["supporting_facts"]:
        if sf[0] in sf_titles:
            sf_titles[sf[0]].append(sf[1])
        else:
            sf_titles[sf[0]] = [sf[1]]

    for i, (title, lines) in enumerate(instance['context']):
        sf_indices = [-1]
        if title in sf_titles:
            sf_indices = sf_titles[title]
            if sf_only:
                lines = [lines[s] for s in sf_indices if s<len(lines)]

        if lowercase:
            lines = [line.lower() for line in lines]
            title = title.lower()

        title = "{0} {1}".format("<title>", title)
        title_ids = tokenizer.encode_plus(title)

        full_context = "".join(lines)
        max_length = max_passage_len - len(title_ids["input_ids"]) - 2

        full_context_ids, line_lengths = get_token_encodings(lines, tokenizer, max_length, add_sent_ends, lowercase)

        context_info.append({"title_text": title, "text": full_context,
                             "title_tokens": title_ids["input_ids"], "tokens": full_context_ids,
                             "sentence_offsets": line_lengths,
                             "sf_indices": sf_indices
                            })
    return context_info


def get_token_encodings(lines, tokenizer, max_length, add_sent_ends, lowercase):
    line_lengths, offset = [], 0
    full_context_ids = [tokenizer.convert_tokens_to_ids("<paragraph>")]
    sent_start_tok, sent_end_tok = tokenizer.convert_tokens_to_ids(["<sent>", "</sent>"])
    if isinstance(lines, str):
        lines = [lines]
    for line in lines:
        if lowercase:
            line = line.lower()

        line_ids = tokenizer.encode_plus(line)["input_ids"]

        word_cnt = len(line_ids) + offset + 1 if add_sent_ends else len(line_ids) + offset
        if word_cnt > max_length:
            ids_to_conc = line_ids[:max_length - len(full_context_ids) - 1] + [sent_end_tok] \
                if add_sent_ends else line_ids[:max_length - len(full_context_ids)]
            full_context_ids += ids_to_conc
            word_cnt = max_length - 1
            line_lengths.append(word_cnt)
            break
        line_lengths.append(word_cnt)
        full_context_ids += line_ids + [sent_end_tok] if add_sent_ends else line_ids
        offset += len(line_ids) + 1 if add_sent_ends else len(line_ids)
    return full_context_ids, line_lengths

def get_dataset(logger, dataset, dataset_cache, dataset_path, split='train', mode='train', randomize=False, use_cache=True):
    dataset_cache = dataset_cache + split + '_' + dataset.__class__.__name__ + '_' + dataset.tokenizer.__class__.__name__
    if use_cache and dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        data = torch.load(dataset_cache)
        if mode == "train" and randomize:
            random.shuffle(data)
        logger.info("Number of instance in {0}: {1}".format(split, len(data)))
        return data

    dataset_path = "%s%s.json" % (dataset_path, split)

    if "hotpot" in dataset.__class__.__name__.lower():
        all_instances = get_hotopot_instances(dataset, dataset_path, mode)
    elif "wikihop" in dataset.__class__.__name__.lower():
        all_instances = get_hotopot_instances(dataset, dataset_path, mode)

    if dataset_cache:
        torch.save(all_instances, dataset_cache)

    logger.info("Dataset cached at %s", dataset_cache)
    logger.info("Number of instance in {0}: {1}".format(split, len(all_instances)))
    return all_instances

def get_hotopot_instances(dataset, dataset_path, mode):
    all_instances = []
    for inst in tqdm(json.load(open(dataset_path))):
        try:
            inst['mode'] = mode
            new_inst = dataset.get_instance(inst)
            if new_inst is not None:
                if isinstance(new_inst, list):
                    for nw in new_inst:
                        all_instances.append(nw)
                else:
                    all_instances.append(new_inst)

        except Exception:
            traceback.print_exc()
            print(inst["_id"])

    return all_instances

def get_data_loaders(dataset, include_train, lazy, use_cache=True):
    logger = dataset.logger
    args = dataset.args
    datasets_raw = {}
    if include_train:
        logger.info("Loading training data")
        datasets_raw['train'] = get_dataset(logger, dataset, args.dataset_cache, args.dataset_path,
                                            args.train_split_name, mode='train', use_cache=use_cache)
    logger.info("Loading validation data")
    datasets_raw['valid'] = get_dataset(logger, dataset, args.dataset_cache, args.dataset_path,
                                        args.dev_split_name, mode='valid', use_cache=use_cache)

    logger.info("Build inputs and labels")

    if lazy:
        if include_train:
            train_dataset = LazyCustomDataset(datasets_raw['train'], dataset, mode='train')
        valid_dataset = LazyCustomDataset(datasets_raw['valid'], dataset, mode='valid')

    else:
        datasets = {
            "train": defaultdict(list),
            "valid": defaultdict(list)
        }
        for dataset_name, dataset_split in datasets_raw.items():
            for data_point in dataset_split:
                instance = dataset.build_segments(data_point)
                for input_name in instance.keys():
                    datasets[dataset_name][input_name].append(instance[input_name])


        logger.info("Pad inputs and convert to Tensor")
        tensor_datasets = {"train": [], "valid": []}
        for dataset_name, dataset_instances in datasets.items():
            if len(dataset_instances) > 0:
                tensor_datasets[dataset_name] = dataset.pad_and_tensorize_dataset(dataset_instances, mode=dataset_name)

        if include_train:
            train_dataset = TensorDataset(*tensor_datasets["train"])
        valid_dataset = TensorDataset(*tensor_datasets["valid"])

    logger.info("Build train and validation dataloaders")
    outputs, metadata = [], []
    if include_train:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        outputs += [train_loader, train_sampler]

        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.predict_batch_size)
        outputs += [valid_loader, valid_sampler]
    else:
        valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.predict_batch_size, shuffle=False)
        outputs += [valid_loader, valid_sampler]

    return outputs

class LazyCustomDataset(Dataset):
    def __init__(self, instances, dataset, mode='train'):
        self.instances = instances
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.mode = mode

    def __getitem__(self, index):
        new_item = self.dataset.build_segments(self.instances[index])
        tensor_instance = self.dataset.pad_and_tensorize_dataset(new_item, self.mode)
        return tuple(tensor_instance)

    def __len__(self):
        return len(self.instances)