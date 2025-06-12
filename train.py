#!/usr/bin/env python3
# coding=UTF-8
version='0.2'
import os
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict, concatenate_datasets, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch
import re
import numpy
import random
import json
import sys
import argparse
import pathlib
import zipfile
from transformers.pytorch_utils import Conv1D
# transformers
import huggingface_hub
try:
    import readtoken
except ModuleNotFoundError:
    print("To download some data sets from HuggingFace, provide your "
        "read token in readtoken.py (not found), or log in with "
        "notebook_login()")
debug=False
# self.token=readtoken.token
# import pushtoken
class Data:
    def load_dbd_from_disk(self):
        #This should be already constrained by language and model processor:
        if os.path.exists(self.fqdatasetname_processed):
            self.dbd=load_from_disk(self.fqdatasetname_processed)
            # with open(self.name_processed,'r') as d:
            #     self.dbd=d.read()
            if hasattr(self,'dbd') and isinstance(self.dbd,DatasetDict):
                # and self.dbd
                self.dataset_prepared=True
                print(f"Found database {self.datasetprettyname} "
                        f"({self.datasetname_processed}) on disk: "
                        f"\n{self.fqdatasetname_processed}")
                # print(f"Columns: {self.dbd.num_columns}")
                print(f"Rows: {self.dbd.num_rows}")
                # print(f"shape: {self.dbd.shape}")
                return True
        else:
            print(f"Local data {self.datasetname_processed} not found")
    def load_csv(self):
        """self.data_files here should be a list of compressed files that you
        want to combine into one split, to be split between train and test
        splits according to self.proportion_to_train_with.

        Here 'audiofolder' loading takes only a compressed file (or folder),
        containing
            -audio files
            -metadata.csv: This file contains
                -an obligatory 'file_name' column, with a row for each file
                -other columns which are imported. Here we use 'sentence'
                    for transcription, in line with general usage.
        For each row in metadata.csv, the audio referenced by 'file_name' is
        cast as Audio() and placed into an 'audio' column, complete with 'path',
        'array', and 'sampling_rate' values.
        """
        # print(f"Using cache dir {self.dataset_dir}")
        print(f"Looking for {self.datasetprettyname} in "
            f"{', '.join(self.data_files)} (in {self.data_file_location})")
        d=load_dataset('audiofolder',
                            data_files=self.data_files
                            ).cast_column("audio", Audio(sampling_rate=16000)
                            ).select_columns(self.columns)
        print(f"unsplit data loaded. Row 1 of {d['train'].num_rows}: "
                f"{d['train'][0]}")
        #for debugging:
        # print(d)
        # for n in range(d['train'].num_rows):
        #     print(f"Row {n} of {d['train'].num_rows}: {d['train'][n]}")
        # exit()
        if not hasattr(self,'proportion_to_train_with'):
            self.proportion_to_train_with=.9 #default to 90% training, 10% test
        self.dbd=d['train'].train_test_split(
                                    train_size=self.proportion_to_train_with)
        rows_to_show=1
        # rows_to_show=self.dbd['train'].num_rows #for more detailed debug
        for split in ['train', 'test']:
            # print(f"{split} data loaded: (type: {type(self.dbd[split])})")
            for n in range(rows_to_show):
                print(f"{split} row {n} of {self.dbd[split].num_rows}: "
                        f"{self.dbd[split][n]}")
        self.dataset_prepared=False
        print(f"Loaded database {self.datasetprettyname} with "
                f"{self.language['name']} data "
                f"({self.fqdatasetname}) from CSV file, "
                f"Rows: {self.dbd.num_rows}")
        # print(f"first row: {self.dbd['train'][0]}")
    def load_dbd(self):
        self.dbd = DatasetDict()
        splits=list(map(lambda x:x+f'[:{self.max_data_rows}]',self.data_splits))
        print(f"Using data splits {splits}")
        print(f"Using cache dir {self.dataset_dir} ({self.fqdatasetname} not found or refreshing it)")
        self.dbd["train"] = load_dataset(self.fqdatasetname, #This should be 1st
                                    self.language['mcv_code'],
                                    split='+'.join(splits),
                                    token=readtoken.token,
                                    trust_remote_code=True,
                                    cache_dir=self.dataset_dir,
                                    ).select_columns(self.columns)
        self.dbd["test"] = load_dataset(self.fqdatasetname,
                                    self.language['mcv_code'],
                                    split="test",
                                    token=readtoken.token,
                                    trust_remote_code=True,
                                    cache_dir=self.dataset_dir,
                                    ).select_columns(self.columns)
        self.dataset_prepared=False
        print(f"Found database {self.datasetprettyname} with "
                f"{self.language['name']} data "
                f"({self.fqdatasetname}) online/cached")
        print(f"Rows: {self.dbd.num_rows}")
    def load_dbd_from_all(self):
        """If you don’t provide a split argument to datasets.load_dataset(),
        this method will return a dictionary containing a datasets for each
        split in the dataset."""
        print(f"Using cache dir {self.dataset_dir}")
        self.dbd = load_dataset(self.fqdatasetname,
                            self.language['mcv_code'],
                            token=readtoken.token,
                            # cache_dir=self.dataset_dir,
                            trust_remote_code=True,
                            cache_dir=self.dataset_dir,
                            ).select_columns(self.columns)
        #combine splits we want:
        self.dbd["train"]=concatenate_datasets([dbd["train"], dbd["validation"]])
        # remove splits we don't want:
        for k in [i for i in self.dbd if i not in ["train","test"]]:
            del self.dbd[k]
        self.dataset_prepared=False
        print(f"Found database {self.datasetprettyname} ({self.fqdatasetname})")
        print(f"Rows: {self.dbd.num_rows}")
    def browse_dataset(self, num_examples=10):
        dataset=self.dbd["train"]
        assert num_examples <= len(dataset), "Can't pick more elements than "
        "there are in the dataset."
        picks = []
        for _ in range(num_examples):
            pick = random.randint(0, len(dataset)-1)
            while pick in picks:
                pick = random.randint(0, len(dataset)-1)
            picks.append(pick)
        print(dataset[picks])
    def show_dimensions_preprocessed(self):
        if self.dataset_prepared:
            return
        rand_int = random.randint(0, len(self.dbd['train'])-1)
        print("Target text:", self.dbd['train'][rand_int][self.transcription_field])
        print("Input array shape:", self.dbd['train'][rand_int]["audio"]["array"].shape)
        print("Sampling rate:", self.dbd['train'][rand_int]["audio"]["sampling_rate"])
    def show_dimensions_processed(self):
        if not self.dataset_prepared:
            return
        rand_int = random.randint(0, len(self.dbd['train'])-1)
        print("Target text:", self.dbd['train'][rand_int].keys())
        print("Target text:", self.dbd['train'][rand_int]["labels"])
        print("Input features:", self.dbd['train'][rand_int]["input_features"])
        print("Input length:", self.dbd['train'][rand_int]["input_length"])
    def process_audio(self):
        self.downsample_audio()
    def downsample_audio(self):
        """Think about relationship wtih Model.prepare_dataset"""
        self.dbd=self.dbd.cast_column("audio", Audio(sampling_rate=16000))
    def to_lower_case(self,batch):
        # remove special characters
        batch[self.transcription_field] = batch[self.transcription_field].lower()
        return batch
    def do_to_lower_case(self):
        self.dbd=self.dbd.map(self.to_lower_case)
    def remove_special_characters(self,batch):
        # remove special characters
        batch[self.transcription_field] = re.sub(self.chars_to_remove_regex, '',
                                batch[self.transcription_field]).lower()

        return batch
    def do_remove_special_characters(self):
        self.chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]'
        self.dbd=self.dbd.map(self.remove_special_characters)
    def remove_latin_characters(self,batch):
        batch[self.transcription_field] = re.sub(r'[a-z]+', '', batch[self.transcription_field])
        return batch
    def do_remove_latin_characters(self):
        self.dbd=self.dbd.map(self.remove_latin_characters)
    def load_chars_from_file(self):
        return False
        #This should ultimately take a list of characters (char_list)
        # from some file, and make something of the form
        self.vocab_dict = {v:k for k,v in enumerate(sorted(char_list))}
    def extract_all_chars(self,batch):
        all_text = " ".join(batch[self.transcription_field])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}
    def do_extract_all_chars(self):
        self.vocab=DatasetDict(self.dbd.copy()) #otherwise .copy() -> dict
        self.vocab=self.vocab.map(self.extract_all_chars,
                                batched=True,
                                batch_size=-1,
                                keep_in_memory=True,
                                remove_columns=self.vocab.column_names["train"])
        # CTC calls this a vocab list, but the vocabulary items are characters!
        char_list = list(set(self.vocab["train"]["vocab"][0]) |
                        set(self.vocab["test"]["vocab"][0]))
        print(self.dbd["train"][0][self.transcription_field])
        print(f"char_list ({len(char_list)}):", char_list)
        self.vocab_dict = {v:k for k,v in enumerate(sorted(char_list))}
        print(f"self.vocab_dict ({len(self.vocab_dict)}):",self.vocab_dict)
    def make_vocab_file(self):
        if not self.load_chars_from_file():
            self.do_extract_all_chars()
        #use defaults from https://huggingface.co/transformers/v4.5.1/model_doc/wav2vec2.html#wav2vec2ctctokenizer
        if True: #I need to think if this is a good idea:
            self.vocab_dict["|"] = self.vocab_dict[" "]#"make 'sp' more visible"
            del self.vocab_dict[" "]
        self.vocab_dict["<unk>"] = len(self.vocab_dict)
        self.vocab_dict["<pad>"] = len(self.vocab_dict)
        len(self.vocab_dict)
        with open(f"vocab_{self.language['iso']}.json", 'w') as vocab_file:
            json.dump(self.vocab_dict, vocab_file)
    def cleanup(self):
        if hasattr(self,'tmp') and os.path.exists(self.tmp):
            for file in os.listdir(self.tmp):
                os.remove('/'.join([self.tmp,file]))
            os.rmdir(self.tmp)
    def __init__(self,**kwargs):
        for k in kwargs:
            if debug:
                print(k,kwargs[k])
            setattr(self,k,kwargs[k])
        self.columns=['audio',self.transcription_field]
        if kwargs.get('refresh_data') or not self.load_dbd_from_disk():
            if self.dataset_code == 'csv':
                self.load_csv()
            else:
                self.load_dbd()
            self.process_audio()
            if kwargs.get('no_special_characters'):
                self.do_remove_special_characters()
            if kwargs.get('no_capital_letters'):
                self.do_to_lower_case()
            if kwargs.get('no_latin_characters'):
                self.do_remove_latin_characters()
            if kwargs.get('make_vocab'):
                self.make_vocab_file()
        #save to disk is after model tokenization
class Processor():
    def push_to_hub(self,repo_name):
        import pushtoken
        print(f"Pushing processor to {repo_name} repo (not yet!)")
        # self.processor.push_to_hub(repo_name,token=pushtoken.token,
        #                             hub_private_repo=self.hub_private_repo)
    def make_tokenizer(self):
        #this should just download or use from cache
        print("Downloading tokenizer or using from cache.")
        self.tokenizer = self.tokenizer_fn.from_pretrained(self.fqmodelname_hf,
                                            # language=self.languagename,
                                            task="transcribe",
                                            # cache_dir=self.cache_dir
                                            )
    def prepare_dataset(self,batch):
        """Think about relationship with Data.downsample_audio"""
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]
        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(audio["array"],
                        sampling_rate=audio["sampling_rate"]).input_features[0]
        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch[self.transcription_field]).input_ids
        return batch
    def do_prepare_dataset(self,dataset):
        if dataset.dataset_prepared:
            print("Dataset seems to have been prepared ealier.")
            return
        #I.e., unless previously loaded, etc
        print(f"Processing Dataset {dataset.datasetprettyname}: "
            f"{dataset.dbd.column_names}, rows:{dataset.dbd.num_rows}")
        # print(dataset.dbd['train'][0]["audio"])
        dataset.dbd = dataset.dbd.map(self.prepare_dataset,
            remove_columns=dataset.dbd.column_names["train"],
            # fn_kwargs={"cache_dir": "/media/kentr/hfcache/datasets"},
            num_proc=count_cpus())
        print(f"Going to save to disk as {dataset.datasetname_processed}: "
                f"{dataset.dbd.column_names}, rows:{dataset.dbd.num_rows}")
        dataset.dbd.save_to_disk(dataset.fqdatasetname_processed)
    def compute_metrics(self,pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        errorrate = 100 * self.metric.compute(predictions=pred_str,
                                                references=label_str)

        return {"errorrate": errorrate}
    def __init__(self,**kwargs):
        for k in kwargs:
            if debug:
                print(k,kwargs[k])
            setattr(self,k,kwargs[k])
        kwargs={}
        if hasattr(self,'cache_dir'):
            kwargs['cache_dir']=self.cache_dir
        # if self.fqmodelname_hf:
        #     fq=self.fqmodelname_hf
        # else:
        #     print("No fully qualified modelname on HF?")#fq=self.modelname
        fq=self.fqbasemodelname #This class is only called when absent online
        self.feature_extractor = self.feature_extractor_fn.from_pretrained(fq)
        # kwargs['languagename']=self.languagename
        self.make_tokenizer()
        # return
        self.processor = self.processor_fn(
                                feature_extractor=self.feature_extractor,
                                tokenizer=self.tokenizer)
        self.processor.save_pretrained(self.fqmodelname_loc)
class BertProcessor(Processor):
    def verify_json(self,json_file):
        defaults=set(['<unk>','<pad>','|'])
        with open(json_file, 'r') as vocab_file:
            d=json.load(vocab_file)
            defaults_present=defaults&set(d)
            if len(defaults_present) == len(defaults):
                print(f"all defaults ({defaults}) found in json file.")
                with open('vocab.json', 'w') as f:
                    json.dump(d,f)
            else:
                print(f"json file missing default value(s) {defaults-set(d)}")
    def make_tokenizer(self):
        print("Building tokenizer from (local) json file.")
        json_file=f"vocab_{self.language['iso']}.json"
        try:
            self.verify_json(json_file)
        except FileNotFoundError as e:
            print(f"It looks like the json file didn't get made; "
                "be sure to set make_vocab=True in data load ({e})")
            exit()
        self.tokenizer = self.tokenizer_fn.from_pretrained('./',
                                        tokenizer_class= 'Wav2Vec2CTCTokenizer')
    def prepare_dataset(self,batch):
        audio = batch["audio"]
        batch["input_features"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(batch["input_features"])
        batch["labels"] = self.processor(text=batch[self.transcription_field]).input_ids
        return batch
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
class BaseModel():
    def get_peftOK_layer_names(self):
        layer_names = []
        layer_types = []
        lora_modules=(torch.nn.Linear,
                        # torch.nn.Embedding,
                        # torch.nn.Conv2d,
                        # torch.nn.Conv3d,
                        # Conv1D,
                        # torch.nn.MultiheadAttention
                        # # torch.nn.Conv1d,
                    )
        # Recursively visit all modules and submodules
        for name, module in self.model.named_modules():
            # Check if the module is an instance of the specified layers
            if isinstance(module, lora_modules):
                # model name parsing
                # print('found', name, module)
                # print(name.split('.')[-1])
                layer_types.append(type(module))
                layer_names.append(name.split('.')[-1])
        print('peftOK_layer_names:',list(set(layer_names)))
        print('peftOK_layer_types:',list(set(layer_types)))
        return list(set(layer_names))

    def use_lora(self):
        # /home/kentr/bin/raspy/newASR/env/lib/python3.11/site-packages/peft/tuners/tuners_utils.py:550: UserWarning: Model with `tie_word_embeddings=True` and the tied_target_modules=['model.decoder.embed_tokens'] are part of the adapter. This can lead to complications, for example when merging the adapter or converting your model to formats other than safetensors. See for example https://github.com/huggingface/peft/issues/2018.
        self.check_for_input_embeddings_method()
        lora_config = LoraConfig(
                                    #per https://discuss.huggingface.co/t/unexpected-keywork-argument/91356:
                                    # task_type=TaskType.SEQ_2_SEQ_LM,
                                    task_type=TaskType('SEQ_2_SEQ_LM'),#'automatic-speech-recognition',
                                    inference_mode=False,
                                    r=8,
                                    lora_alpha=32,
                                    lora_dropout=0.1,
                                    target_modules=self.get_peftOK_layer_names(),
                                    # save_embedding_layers=True #necessary when changing vocab
                                )
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        # model = get_peft_model(model, peft_config)
        self.model = get_peft_model(self.model, lora_config)
        # This should be used to load a file saved earlier:
        # self.model = PeftModel.from_pretrained(model=self.model,
        #                                         model_id=<location on file system>
        #                         is_trainable=True)
        self.model.print_trainable_parameters()
        self.did_lora=True
    def use_quantization(self):
        from transformers import pipeline

        # Load pipeline with quantization
        pipe = pipeline("text-classification", model="bert-base-uncased", device=0)

        # Apply quantization
        pipe.model = torch.quantization.quantize_dynamic(
            pipe.model, {torch.nn.Linear}, dtype=torch.qint8
        )
    def donothing(self):
        pass
    def check_for_input_embeddings_method(self):
        if not hasattr(self.getmodel_fn,'get_input_embeddings'):
            self.getmodel_fn.get_input_embeddings=self.donothing
    def get_model(self,**kwargs):
        self.model = self.getmodel_fn.from_pretrained(self.fqbasemodelname,
                                                        **kwargs)
        try:
            if self.model.generation_config:
                self.model.generation_config.language = self.language['name']
                self.model.generation_config.task = "transcribe"
                self.model.generation_config.forced_decoder_ids = None
        except Exception as e:
            print("sounds like not all the config went to this kind of model "
                f"({e})")
        self.did_lora=False
        if kwargs.get('quantization_config'):
            self.did_quant=True
    # def model_fns(self):
    #     self.metric = evaluate.load(self.metric_name, cache_dir=self.cache_dir)
    def __init__(self,**kwargs):
        #fqmodelname,getmodel_fn,languagename
        #fqmodelname inlcudes parent repo
        # kwargs['fqname']=kwargs['fqmodelname']
        kwargstogetmodel=[
                        'pretrained_model_name_or_path',
                        'config',
                        'cache_dir',
                        'ignore_mismatched_sizes',
                        'force_download',
                        'local_files_only',
                        'token',
                        'revision',
                        'use_safetensors',
                        'weights_only',
                        'vocab_size',
                        'attention_dropout',
                        'hidden_dropout',
                        'feat_proj_dropout',
                        'mask_time_prob',
                        'layerdrop',
                        'ctc_loss_reduction',
                        # 'add_adapter',
                        ]
        nonmodelkwargs=['getmodel_fn',
                        'language',
                        'lang_code',
                        'lora',
                        'quant',
                        'metric_name',
                        'modelprettyname',
                        'pad_token_id',
                        'basemodelprettyname',
                        'fqbasemodelname',
                        'push_to_hub'
                        ]
        # assert not set(nonmodelkwargs)&set(kwargstogetmodel)
        overlap=set(kwargstogetmodel)&set(nonmodelkwargs)
        remaining=set(kwargs)-set(kwargstogetmodel)-set(nonmodelkwargs)
        try:
            assert not overlap
        except AssertionError as e:
            print("Found overlapping model kwargs (remove?):",
                    overlap)
            exit()
        try:
            assert not remaining
        except AssertionError as e:
            print("Found remaining model kwargs (add them?):",
                    remaining)
            exit()
        for k in kwargs.copy(): #b/c pop
            if k in kwargstogetmodel:
                if debug:
                    print(k,kwargs[k],"(to get_model())")
                setattr(self,k,kwargs[k]) #leave in kwargs
            else:
                if debug:
                    print(k,kwargs[k],"(just for this BaseModel object)")
                setattr(self,k,kwargs.pop(k))
        self.get_model(**kwargs)
        if getattr(self,'lora',False):
            self.use_lora()
            print('LoRA:',self.did_lora)
            if not self.did_lora:
                log.error("LoRA didn't work!")
                exit()
        if isinstance(self.model, torch.nn.Module):
            print(f"Model {self.basemodelprettyname} ({self.fqbasemodelname}) "
                "loaded for training on "
                f"{self.language['name']} [{self.language['iso']}] data")
        # # Apply quantization
        # pipe.model = torch.quantization.quantize_dynamic(
        #     pipe.model, {torch.nn.Linear}, dtype=torch.qint8
        # )
        # self.model
class Training():
    def training_arguments(self,**kwargs):
        """This is from whisper recipie, as is"""
        # per_device_train_batch_size:
        # pdtbs=16
        if not kwargs.get('per_device_train_batch_size'):
            kwargs['per_device_train_batch_size']=16
        # increase by 2x per 2x decrease in batch size:
        #gradient_accumulation_steps:
        gas=16//kwargs.get('per_device_train_batch_size')
        if not torch.cuda.is_available():
            kwargs['dataloader_pin_memory']=False
        return self.training_args_fn(
                                # change to a repo name of your choice:
                            output_dir=self.fqmodelname_loc,
                            # per_device_train_batch_size=pdtbs,
                            gradient_accumulation_steps=gas,
                            warmup_steps=500,
                            # Not for LoRA:
                            # gradient_checkpointing=True,
                            fp16=True,
                            # eval_strategy="steps",
                            # evaluation_strategy
                            report_to=["tensorboard"],
                            load_best_model_at_end=True,
                            #pick from 'eval_loss', 'eval_errorrate', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch'
                            metric_for_best_model='eval_loss',
                            greater_is_better=False,
                            remove_unused_columns=False,
                            **kwargs)
    def training_functions(self):
        """This appears to not be used"""
        self.processor = self.processor_fn.from_pretrained(self.model.fqname,
                                            language=self.language['name'],
                                            task="transcribe",
                                            cache_dir=self.cache_dir
                                            )
        self.data_collator = self.data_collator_fn(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            )
    def train(self):
        """
        /home/kentr/bin/raspy/newASR/env/lib/python3.11/site-packages/torch/utils/data/dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
        warnings.warn(warn_msg)

        Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.

        `use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`...
        """
        self.trainer.train()
    def push(self):
        self.trainer.push_to_hub(**self.push_kwargs())
        self.tokenizer.push_to_hub(self.modelname, **self.push_kwargs())
    def push_kwargs(self):
        import pushtoken
        return {
                    "dataset_tags": self.db.name,
                    # a 'pretty' name for the training dataset
                    "hub_private_repo": self.hub_private_repo,
                    "dataset": self.db.datasetprettyname,
                    "dataset_args": f"config: {self.language['mcv_code']}, split: test",
                    "language": self.language['iso'],
                    # a 'pretty' name for your model
                    "model_name": f"{self.modelname} "
                                f"{self.language['iso']} - {self.language['name']}",
                    "finetuned_from": self.fqbasemodel,
                    "tasks": "automatic-speech-recognition",
                    "token": pushtoken.token
                }
    def demo(self):
        d=Demo(self)
    def __init__(self,**kwargs):
        not_for_training_arguments_kwargs=['fqbasemodelname',#these are for this module
                                            'fqmodelname_loc',
                                            'model',
                                            'processor',
                                            'data',
                                            'data_collator',
                                            'compute_metrics',
                                            'modelname',
                                            'data_collator_fn',
                                            'training_args_fn',
                                            'trainer_fn',
                                            ]
        for_training_arguments_kwargs=[#just what huggingface wants
            'eval_strategy',
            'save_strategy',
            'per_device_train_batch_size',
            'learning_rate',
            'save_steps',
            'eval_steps',
            'logging_steps',
            'save_total_limit',
            'num_train_epochs',
            ]
        overlap=set(for_training_arguments_kwargs)&set(
                                            not_for_training_arguments_kwargs)
        remaining=set(kwargs)-set(for_training_arguments_kwargs)-set(
                                            not_for_training_arguments_kwargs)
        try:
            assert not overlap
        except AssertionError as e:
            print("Found overlapping training kwargs (remove?):",
                    overlap)
            exit()
        try:
            assert not remaining
        except AssertionError as e:
            print("Found remaining training kwargs (add them?):",
                    remaining)
            exit()
        for k in kwargs.copy(): #b/c pop
            if k in for_training_arguments_kwargs:
                if debug:
                    print(k,kwargs[k],"(to training_arguments())")
                setattr(self,k,kwargs[k]) #leave in kwargs
            else:
                if debug:
                    print(k,kwargs[k],"(just for this Training object)")
                setattr(self,k,kwargs.pop(k))
        # for k in [i for i in kwargs if '_fn' in i
        #                             or i.startswith('model')
        #                             or i.startswith('data')
        #                             or i.startswith('processor')
        #                             or i.startswith('compute')
        #         ]:
        #     setattr(self,k,kwargs.pop(k))
        self.trainer = self.trainer_fn(model=self.model,
                            args=self.training_arguments(**kwargs),
                            train_dataset=self.data["train"],
                            eval_dataset=self.data["test"],
                            data_collator=self.data_collator,
                            compute_metrics=self.compute_metrics,
                            # tokenizer=self.processor.feature_extractor #not tokenizer
                            processing_class=self.processor.feature_extractor, #not tokenizer
                            )
class Demo(object):
    def transcribe_module(self,audio):
        return self.inferer(audio,show_standard=True)
    def transcribe_pipe(self,audio):
        return self.pipe(audio)["text"]
    def __init__(self,names):
        from transformers import pipeline
        import gradio as gr
        import infer
        self.inferer=infer.Infer(names.fqmodelname_loc)
        self.pipe = pipeline(model=names.fqmodelname_loc,
                            model_kwargs={"cache_dir": names.cache_dir},
                            tokenizer=names.fqmodelname_loc,
                            task='automatic-speech-recognition',
                            )
        inputs1=gr.Audio(sources=['microphone', 'upload'], type="filepath"),
        inputs2=gr.Audio(sources=['microphone', 'upload'], type="filepath"),
        iface_pipe = gr.Interface(
            fn=self.transcribe_pipe,
            inputs=inputs1,
            outputs="text",
            title=f"Automatic Speech Recognition (ASR) {names.language['name']}",
            description=(f"Realtime demo for {names.language['name']} speech "
                        "recognition using a fine-tuned "
                        f"{names.modelprettyname} model."),
            )
        iface_module = gr.Interface(
            fn=self.transcribe_module,
            inputs=inputs2,
            outputs="text",
            title=f"Automatic Speech Recognition (ASR) {names.language['name']}",
            description=(f"Realtime demo for {names.language['name']} speech "
                        "recognition using a fine-tuned "
                        f"{names.modelprettyname} model."),
            )
        app = gr.TabbedInterface(interface_list=[iface_pipe, iface_module],
                         tab_names = ["pipe", "module"])
        app.launch()
@dataclass
class DataCollatorCTCWithPadding:
    from transformers import Wav2Vec2BertProcessor
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
class Nomenclature():
    """This class calculates and stores names for various things, based on
    provided parameters"""
    def tuned_repos_loc_hf(self):
        if self.fqmodelname_hf:
            return (self.fqmodelname_loc,self.fqmodelname_hf)
        else:
            return (self.fqmodelname_loc) #only send what's there
    def processorkwargs(self):
        attrs=['tokenizer_fn',
            'feature_extractor_fn',
            'processor_fn',
            'transcription_field',
            'metric',
            'metric_name',
            'fqmodelname_hf',
            'fqmodelname_loc',
            'fqbasemodelname',
            'modelname',
            'cache_dir',
            'language'
        ]
        return {a:getattr(self,a) for a in attrs if hasattr(self,a)}
    def datakwargs(self):
        attrs=['mcv_code',
                'dataset_code',
                'data_files',
                'data_file_location',
                'language',
                'dataset_dir',
                'transcription_field',
                'max_data_rows',
                'refresh_data',
                'no_capital_letters',
                'no_special_characters',
                'make_vocab',
                'fqdatasetname',
                'datasetprettyname',
                'data_splits',
                'datasetname_processed',
                'fqdatasetname_processed',
                'proportion_to_train_with',
                'hub_private_repo'
            ]
        return {a:getattr(self,a) for a in attrs if hasattr(self,a)}
    def modelkwargs(self):
        attrs=['fqbasemodelname',
                'modelprettyname',
                'basemodelprettyname',
                'getmodel_fn',
                'cache_dir',
                'attention_dropout',
                'hidden_dropout',
                'feat_proj_dropout',
                'mask_time_prob',
                'layerdrop',
                'ctc_loss_reduction',
                'language',
                'iteration',
                'lora',
                'quant'
                ]
        return {a:getattr(self,a) for a in attrs if hasattr(self,a)}
    def trainingkwargs(self):
        attrs=['fqbasemodelname',
                'fqmodelname_loc',
                'data_collator_fn',
                'training_args_fn',
                'trainer_fn',
                'data_collator_fn',
                'training_args_fn',
                'trainer_fn',
                'per_device_train_batch_size',
                'eval_strategy',
                'save_strategy',
                'compute_metrics',
                'learning_rate',
                'save_steps',
                'eval_steps',
                'logging_steps',
                'save_total_limit',
                'num_train_epochs',
                ]
        return {a:getattr(self,a) for a in attrs if hasattr(self,a)}
    def init_languages(self):
        self.languages={'gnd':{'mcv_code':'gnd', 'iso':'gnd', 'name':'Zulgo'},
                        "yo":{'mcv_code':'yo', 'iso':'yor', 'name':"Yoruba"},
                        "sw":{'mcv_code':'sw', 'iso':'swh', 'name':"Swahili"},
                        "ig":{'mcv_code':'ig', 'iso':'ibo', 'name':"Igbo"},
                        "rw":{'mcv_code':'rw', 'iso':'kin', 'name':"Kinyarwanda"},
                        "lg":{'mcv_code':'lg', 'iso':'lug', 'name':"Luganda"},
                        }
    def setlang(self,**kwargs):
        self.init_languages()
        language_iso=kwargs.get('language_iso')
        if language_iso:
            try:
                self.language=self.languages[language_iso]
            except KeyError:
                onelangperline='\n'.join([k+":\t"+str(self.languages[k]).strip('{}') for k in self.languages])
                print(f"language code ‘{language_iso}’ not found; the following "
                     f"are currently set up:\n{onelangperline}"
                      )
                exit()
        else:
            print("You really need to specify your language with an ISO code "
            "(e.g., -l/--language-iso)")
            exit()
    def noparens_dirs(self,x):
        return '_'.join(x.translate(self.parens_dirs_dict).split(' '))
    def dataset_name(self,**kwargs):
        if self.data_file_prefixes and not self.dataset_code:
            self.dataset_code='csv'
        self.parens_dirs_dict=str.maketrans('','',')(')
        d={
            'mcv':"mozilla-foundation/common_voice_17_0",
            }
        if hasattr(self,'data_files'):
            d['csv']=f"Local CSV files ({', '.join(self.data_files)})"
        else:
            d['csv']=f"Local CSV files (provide -d/--data-file-prefix(es))"
        try:
            self.fqdatasetname=d[self.dataset_code]
        except AttributeError:
            print("You need to provide a 'dataset_code'; this impacts "
                "the name of the model, which reflects the dataset "
                "it was trained on (use 'csv' for local CSV files).")
            exit()
        except KeyError:
            oneperline='\n'.join([k+":\t"+str(d[k]).strip('{}') for k in d])
            print(f"dataset code ‘{self.dataset_code}’ not found; "
                    f"the following are currently set up:\n{oneperline}")
            exit()
        if self.dataset_code not in ['csv']:
            self.datasetprettyname=huggingface_hub.dataset_info(
                                                    self.fqdatasetname,
                                                    ).card_data.pretty_name
            self.dataset_configbits=self.fqdatasetname.split('/')
            self.dataset_configbits+=[self.language['mcv_code']]
            self.dataset_configbits+=self.data_splits
        else:
            self.datasetprettyname=f"Local {self.language['name']} data"
            self.dataset_configbits=[self.language['iso']]
            self.dataset_configbits+=self.data_file_prefixes
        if kwargs.get('no_special_characters'):
            self.dataset_configbits+=['nospchars']
        if kwargs.get('no_capital_letters'):
            self.dataset_configbits+=['nocaps']
        if kwargs.get('no_latin_characters'):
            self.dataset_configbits+=['nolatin']
        self.dataset_configbits+=['to_train']+self.fqbasemodelname.split('/')
        self.dataset_configbits+=[self.feature_extractor_fn.__name__,
                                    "processed"]
        self.dataset_configbits=list(map(self.noparens_dirs,
                                            self.dataset_configbits))
        self.datasetname_processed='-'.join(self.dataset_configbits)
        self.fqdatasetname_processed=os.path.join(
                                        self.cache_dir,
                                        self.datasetname_processed)
    def model_name(self,**kwargs):
        """This function names the model to be created through tuning"""
        prettybits=[]
        tuned_bits=[self.basemodelname]
        if kwargs.get('quant'):
            tuned_bits+=['quant']
            prettybits+=['Quantized']
        if kwargs.get('lora'):
            tuned_bits+=['lora']
            prettybits+=['LoRA']
        tuned_bits+=[self.metric_name]
        tuned_bits+=[self.language['iso'],self.language['name']]
        if kwargs.get('increment'):
            tuned_bits+=[str(self.increment)] #in case this is int
        self.modelname='-'.join(tuned_bits)
        if (getattr(self,'data_file_prefixes',None) and
            self.data_file_prefixes[0].split('_')[-1].isdigit()): #test once for all
            rows=str(sum([int(i.split('_')[-1]) for i in self.data_file_prefixes]))
            print(f"using {rows} rows")
        else:
            print("getattr data_file_prefixes:",getattr(self,'data_file_prefixes',None))
            print("int:",int(self.data_file_prefixes[0].split('_')[-1]))
            print("int:",self.data_file_prefixes[0].split('_')[-1].isdigit())
            print("int options:",self.data_file_prefixes)
            rows=''
        self.modelname+=f'_{rows}x{str(self.num_train_epochs)}'
        self.fqmodelname_loc=os.path.join(
                                        self.cache_dir,
                                        self.modelname)
        if kwargs.get('my_hf_login'):
            self.fqmodelname_hf='/'.join([kwargs.get('my_hf_login'),self.modelname])
        else:
            self.fqmodelname_hf=False
        repo_errors=(AttributeError, OSError,
                    huggingface_hub.errors.HFValidationError,
                    huggingface_hub.errors.RepositoryNotFoundError)
        try:
            self.modelprettyname=huggingface_hub.model_info(
                                                    self.fqmodelname_hf,
                                                    ).card_data.pretty_name
        except repo_errors:
            self.modelprettyname=' '.join(self.modelname.split('-')).title()
        try:
            self.basemodelprettyname=huggingface_hub.model_info(
                                                self.fqbasemodelname,
                                                ).card_data.pretty_name
        except repo_errors:
            self.basemodelprettyname=' '.join(self.basemodelname.split('-')).title()
        if prettybits: #These are the ultimate model loaded
            self.modelprettyname+=f" ({' '.join(prettybits)})"
            self.basemodelprettyname+=f" ({' '.join(prettybits)})"
    def check_cache(self):
        if hasattr(self,'cache_dir') and self.cache_dir:
            os.environ['HF_HOME'] = self.cache_dir
        else:
            self.cache_dir = os.environ['HF_HOME'
                                    ] = f"/media/{os.environ['USER']}/hfcache/"
        if not os.path.exists(os.environ['HF_HOME']):
            print("This is set up to run with a cache at "
                    f"{os.environ['HF_HOME']}, "
                    "but it isn't found. Attach it, pass -c/--cache-dir, "
                    "or change the os.environ['HF_HOME'] variable.")
            exit()
        self.dataset_dir=os.path.join(self.cache_dir,'datasets')
        self.hub_model_dir=os.path.join(self.cache_dir,'hub')
    def __init__(self,**kwargs):
        self.setlang(**kwargs)
        for k in kwargs:
            if debug:
                print(k,kwargs[k])
            setattr(self,k,kwargs[k])
        # if kwargs.get('fqmodelname'):
        #     self.modelname=self.fqmodelname.split('/')[-1] #just the name
        if kwargs.get('fqbasemodelname'):
            self.basemodelname=self.fqbasemodelname.split('/')[-1]
        else:
            print("You really need to specify your full model address/name. "
                    "For inference, this is used to find the name of the "
                    "model to use for inference.")
            quit()
        self.check_cache()
        self.model_name(**kwargs)
        if kwargs.get('data_file_prefixes'): #find actual files
            globs=[k for k in [
                '_'.join([self.language['iso'],f'{i}{j}.tar.xz'])
                for i in self.data_file_prefixes
                for j in ['','_*']
            ]
            ]
            self.data_files=[str(i) for j
                        in [pathlib.Path(self.data_file_location).glob(g)
                        for g in globs]
                        for i in j
                        ]
        if self.train:
            self.dataset_name()
class Options(object):
    def has_argv(self):
        """This is needed because colab adds an extra -f root argument, meaning
        that it runs with len(sys.argv) = 3, without any visible user arguments
        So this is true when neither in colab, nor without argv specified."""
        val=not bool('google.colab' in sys.modules
                            or len(sys.argv) == 1)#just exe, no other args
        return val
    def __init__(self):
        self.default_list = [('-l', '--language-iso',
                            {'help':"ISO 639-3 (Ethnologue) code of language",
                            'required':self.has_argv()
                        }),
                        ('-c', '--cache-dir',
                            {'help':"where models and data are stored locally"
                            #Yes, this is redundant, but matches Huggingface use
                            }),
                        ('-p', '--push-to-hub',
                            {'help':"Store model and processor on HuggingFace",
                            'action':'store_true'
                        }),
                        ('--hub-private-repo',
                            {'help':"Store model and processor on HuggingFace "
                                    "privately",
                            'action':'store_true'
                        }),
                        ('--my-hf-login',
                            {'help':"User login for hugging face "
                                        "(for downloads and pushing) ",
                        }),
                        ('-t', '--train',
                            {'help':"Train a new ASR model",
                            'action':'store_true'
                        }),
                        ('--metric-name',
                            {'help':"Name of metric to evaluate ASR (e.g., "
                                "word error rate; WER)",
                            'choices':['wer','cer'],
                            'default':'wer'
                        }),
                        ('--remake_processor',
                            {'help':"Remake Data pre-processor and tokenizer "
                                    "(even if found)",
                            'action':'store_true'
                        }),
                        ('-d', '--data-file-prefix',
                            {'help':"prefix for data archives (multiple OK)",
                            'action':'append',
                            'dest':'data_file_prefixes'
                        }),
                        ('--dataset-code',
                            {'help':"data source (e.g., csv: comma separated, "
                                "mcv: mozilla common voice)",
                            'action':'append'
                        }),
                        ('--data-file-location',
                            {'help':"location of data archives",
                            'default':'./training_data'
                        }),
                        ('-r','--refresh-data',
                            {'help':"Load and process training data even if "
                                "preprocessed data is found",
                            'action':'store_true'
                        }),
                        ('--capital-letters-ok',
                            {'help':"Don't remove Capital Letters from "
                                    "training data",
                            'action':'store_true'
                        }),
                        ('--special-characters-ok',
                            {'help':"Don't remove special characters from "
                                    "training data",
                            'action':'store_true'
                        }),
                        ('--make-vocab',
                            {'help':"make_vocab",
                        }),
                        ('--transcription-field',
                            {'help':"Name of data field containinig "
                                    "transcriptions",
                            'default':'sentence'
                        }),
                        ('--proportion-to-train-with',
                            {'help':"Proportion of data to use for training "
                                "(The remaining will be used for validation)",
                                'default':0.9
                        }),
                        ('-m','--remake_model',
                            {'help':"Remake Model (even if found)",
                            'action':'store_true'
                        }),
                        ('--lora',
                            {'help':"Use Low-Rank Adaptation (LoRA)",
                            'action':'store_true'
                        }),
                        ('--add-adapter',{'help':"Add adapter "
                                                # "Low-Rank Adaptation (LoRA)"
                        ,'action':'store_true'
                        }),
                        ('-q','--quant',
                            {'help':"Use Quantization",
                            'action':'store_true'
                        }),
                        ('--attention-dropout',{'default':0.0}),
                        ('--hidden-dropout',{'default':0.0}),
                        ('--feat-proj-dropout',{'default':0.0}),
                        ('--mask-time-prob',{'default':0.0}),
                        ('--layerdrop',{'default':0.0}),
                        ('--ctc-loss-reduction',
                            {'help':"ctc_loss_reduction",
                            'default':'mean'
                        }),
                        ('-i', '--infer',
                            {'help':"Infer on a model (get text from audio)",
                            'action':'store_true'
                        }),
                        ('-a', '--audio',
                            {'help':"Audio file to infer (convert to text)",
                            'action':'append'
                        }),
                        ('--demo',
                            {'help':"serve a demonstration web page",
                            'action':'store_true'
                        }),
                        ('--debug',
                            {'help':"more output for debugging",
                            'action':'store_true'
                        }),
                        ('-v', '--version', {'action':'version',
                                            'version':f'%(prog)s {version}'
                                        })
                        ]
        if self.has_argv():#'google.colab' in sys.modules: #no sys.argv here
            print("parsing args!")
            self.parse_argv()
        else:
            self.defaults_only()
    def defaults_only(self):
        def sanify_arg(x):
            return x.strip('-').translate(str.maketrans('-','_'))
        self.args={sanify_arg(arg):kwargs['default']
            for *args,kwargs in self.default_list
            for arg in args
            if '--' in arg
            if 'default' in kwargs
        }
        self.args.update({sanify_arg(arg):False #default!
            for *args,kwargs in self.default_list
            for arg in args
            if '--' in arg
            if 'action' in kwargs
            if kwargs['action'] == 'store_true'
        })
    def parse_argv(self):
        parser = argparse.ArgumentParser(
                            prog='ASR_Trainer',
                            description='This module programmatically trains Automatic '
                            'Speech Recognition (ASR) modules, for scalable mass '
                            'production with minimal training data.'
                            '\nUsing various options, one can train, infer and push '
                            'all in the same run, if desired.',
                            )
        for *args,kwargs in self.default_list:
            parser.add_argument(*args,**kwargs)
        self.args = vars(parser.parse_args())
        # This section converts various settings from what makes sense to the
        # user to what the computer uses, especially where default is true,
        # rather than false (unspecified)
        for i in [
            # This should be a list, even if user doesn't think of it this way:
                # ('data_file_prefix','data_file_prefixes'),
                ('special_characters_ok','no_special_characters'),
                ('capital_letters_ok','no_capital_letters')
                ]: #conver first to second
            self.args[i[1]]=self.args.pop(i[0])
        # print(f"Found user args {self.args}")
def compute_metrics_bert(pred):
    """This is outside the class because it needs to be accessable
    even when the processor is loaded, rather than built."""
    pred_logits = pred.predictions
    pred_ids = numpy.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    error = metric.compute(predictions=pred_str, references=label_str)

    return {names.metric_name: error}
def count_cpus():
    #Not sure if this is working; override:
    # self.cpus=10
    # return
    import multiprocessing #not working??!?
    return multiprocessing.cpu_count()

if __name__ == '__main__':
    #for testing with Bert; this is normally in train wrapper:
    from transformers import (Wav2Vec2CTCTokenizer,SeamlessM4TFeatureExtractor,
                                Wav2Vec2BertProcessor,Wav2Vec2BertForCTC,
                                TrainingArguments,Trainer)
    model_type={
                # 'getmodel_fn':AutoModelForCTC, #for basemodels
                'getmodel_fn':Wav2Vec2BertForCTC, #for tuned models
                'tokenizer_fn':Wav2Vec2CTCTokenizer,
                'feature_extractor_fn':SeamlessM4TFeatureExtractor,
                'processor_fn':Wav2Vec2BertProcessor,
                }
    trainer_type={
                'data_collator_fn':DataCollatorCTCWithPadding,
                'training_args_fn':TrainingArguments,
                'trainer_fn':Trainer,
                'learning_rate':5e-5,
                'per_device_train_batch_size':16,
                'save_steps':5,
                # load_best_model_at_end requires the save and eval strategy to match
                'eval_strategy': 'epoch',
                'save_strategy': 'epoch',
                # 'eval_strategy':"steps",
                # 'eval_steps':5,
                'logging_steps':5,
                'save_total_limit':2,
                'num_train_epochs':3,
                'compute_metrics':compute_metrics_bert,
                }
    options=Options()
    options.args.update({
            'language_iso':'gnd',
            'cache_dir':'.',
            'data_file_prefixes':['lexicon_41','examples_300'],
            'train':True
        })
    names=Nomenclature(
        fqbasemodelname="facebook/w2v-bert-2.0",
        **model_type,
        **trainer_type,
        # **options,
        **options.args #pull in default and user settings
        )
    # options={'debug_names':True}
    if options.args.get('debug'):
        for attr in dir(names):
            print(attr,getattr(names,attr))
    data=Data(**names.datakwargs())
    try:
        print(f"Looking for {names.modelname} processor online/cached")
        assert options.get('remake_processor') is False
        processor=names.processor_fn.from_pretrained(
                                                        names.fqmodelname_hf,
                                                        cache_dir=names.cache_dir
                                                        )
                            # feature_extractor=self.feature_extractor,
                            # tokenizer=self.tokenizer)
        if not self.data.dataset_prepared:
            self.data.show_dimensions_preprocessed()
            self.processor.do_prepare_dataset(self.data)
        print(f"Loaded {names.modelname} processor ({names.fqmodelname})")
    except (Exception,AssertionError) as e:
        if isinstance(e,AssertionError):
            e="by request"
        print(f"Remaking {names.modelname} processor ({e})")
        processor=BertProcessor(
                        **names.processorkwargs()
                    )
        data.show_dimensions_preprocessed()
        processor.do_prepare_dataset(data)
        data.cleanup()
        if options.args.get('push_to_hub'):
            processor.push_to_hub(fqmodelname)
        #make this match try results: a transformers object
        processor=processor.processor
    try:
        print(f"Looking for saved {names.modelname} model")
        assert options.args.get('remake_model') is False
        model=names.getmodel_fn.from_pretrained(
                                                names.modelname,
                                                cache_dir=names.cache_dir
                                                )
        print(f"Loaded {names.modelname} model ({names.modelname})")
    except (Exception,AssertionError) as e:
        if isinstance(e,AssertionError):
            e="by request"
        print(f"Remaking {names.modelname} model ({e})")
        model=BaseModel(
                        vocab_size=len(processor.tokenizer),
                        pad_token_id=processor.tokenizer.pad_token_id,
                        **names.modelkwargs()
                        )#fqmodelname=model_fns['fqmodelname'],
        #make this match try results: a transformers object
        model=model.model
    data_collator = names.data_collator_fn(processor=processor, padding=True)
    import evaluate
    metric = evaluate.load(names.metric_name)
    trainer=Training(
                    model= model,
                    processor=processor, #downloads or builds above
                    data=data.dbd,
                    data_collator=data_collator,
                    # compute_metrics=names.compute_metrics_bert,
                    **names.trainingkwargs()
                    )
    trainer.train()
    model.save_pretrained(names.fqmodelname_loc)
    # self.trainer.push()
