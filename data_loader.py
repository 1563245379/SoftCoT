
import os
import json
from typing import Union, Dict
import ast

import pandas as pd
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle


class GSM8KLoader(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()

        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data = json.loads(line)
                    instance = Instance(**data)
                    ds.append(instance)

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = os.path.join(os.path.dirname(__file__), 'data', 'gsm8k')) -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'train_socratic.jsonl'),
                'dev': os.path.join(paths, 'test_socratic.jsonl'),
                'test': os.path.join(paths, 'test_socratic.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class AQuALoader(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path) as f:
            ins_list = json.load(f)

        for ins in ins_list:
            instance = Instance(**ins)
            ds.append(instance)

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = '/path/to/data/dir') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'gsm_style_train.jsonl'),
                'dev': os.path.join(paths, 'gsm_style_dev.jsonl'),
                'test': os.path.join(paths, 'gsm_style_test.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class DULoader(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path) as f:
            ins_list = json.load(f)

        for ins in ins_list:
            instance = Instance(**ins)
            ds.append(instance)

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = '/path/to/data/dir') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'date_understanding_gsm_style.json'),
                'dev': os.path.join(paths, 'date_understanding_gsm_style.json'),
                'test': os.path.join(paths, 'date_understanding_gsm_style.json')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class StrategyQALoader(Loader):

    def __init__(self, train_split=0.8):
        super().__init__()
        self.train_split = train_split

    def _load(self, path: str, is_train=True) -> DataSet:
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        num_train = int(len(dataset) * self.train_split)
        if is_train:
            for ins in dataset[:num_train]:
                ds.append(Instance(**ins))
        else:
            for ins in dataset[num_train:]:
                ds.append(Instance(**ins))
        return ds

    def load(self, paths: Union[str, Dict[str, str]] = '/path/to/data/dir') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'strategyqa_train.json'),
                'dev': os.path.join(paths, 'strategyqa_train.json'),
                'test': os.path.join(paths, 'strategyqa_train.json')
            }

        return DataBundle(datasets={k: self._load(v, is_train=('train' in [k])) for k, v in paths.items()})


class AugASDivLoader(GSM8KLoader):
    def load(self, paths: Union[str, Dict[str, str]] = '/path/to/data/dir') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'aug-train.jsonl'),
                'dev': os.path.join(paths, 'aug-dev.jsonl'),
                'test': os.path.join(paths, 'aug-dev.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


