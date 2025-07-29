#!/usr/bin/env python3
# coding=UTF-8
#This started from a number of https://huggingface.co/blog/ entries on fine tuning
import os
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict, concatenate_datasets, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import get_peft_model, LoraConfig #,TaskType
# from peft import PeftModel #for inference
import torch
import evaluate
import re
import numpy
import random
import json
import sys
import pathlib
import zipfile
from transformers.pytorch_utils import Conv1D
# transformers
import huggingface_hub
try:
    import readtoken
except ModuleNotFoundError:
    print("No read token found; if you have not logged in with "
        "notebook_login(), you will not be able to get data from gated repos")
import options
debug=False
# self.token=readtoken.token
# import pushtoken
def in_venv():
    return sys.prefix != sys.base_prefix
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
        #This indicates data with suprasegmental features
        d = d.filter(lambda x: '+' not in x, input_columns=["sentence"])
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
        if debug:
            for split in ['train', 'test']:
                for n in range(rows_to_show):
                    print(f"{split} row {n} of {self.dbd[split].num_rows}: "
                            f"{self.dbd[split][n]}")
        self.dataset_prepared=False
        print(f"Loaded database {self.datasetprettyname} with "
                f"{self.language['name']} data with {self.dbd.num_rows} rows "                f"from CSV file ({self.fqdatasetname})")
    def load_dbd(self):
        self.dbd = DatasetDict()
        splits=list(map(lambda x:x+f'[:{getattr(self,"max_data_rows","")}]',self.data_splits))
        print(f"Using data splits {splits}")
        test=f'test[:{getattr(self,"max_data_rows","")}]'
        print(f"Using test split {test}")
        print(f"Using cache dir {self.dataset_dir} ({self.fqdatasetname} not found or refreshing it)")
        kwargs={'trust_remote_code':True,
                'cache_dir':self.dataset_dir
                }
        try:
            import readtoken
            kwargs.update({'token':readtoken.token})
        except:
            pass
            # kwargs.update({'use_auth_token':True})
        self.dbd["train"] = load_dataset(self.fqdatasetname, #This should be 1st
                                    self.sister_language['mcv_code'],
                                    split='+'.join(splits),
                                    **kwargs
                                    ).select_columns(self.columns)
        self.dbd["test"] = load_dataset(self.fqdatasetname,
                                    self.sister_language['mcv_code'],
                                    split=test,
                                    **kwargs
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
    def make_vocab_dict(self):
        if not self.load_chars_from_file():
            self.do_extract_all_chars()
        #use defaults from https://huggingface.co/transformers/v4.5.1/model_doc/wav2vec2.html#wav2vec2ctctokenizer
        if True: #I need to think if this is a good idea:
            self.vocab_dict[self.data_tokens[ #"make 'sp' more visible"
                                'word_delimiter_token']] = self.vocab_dict[" "]
            del self.vocab_dict[" "]
        self.vocab_dict[self.data_tokens['unk_token']] = len(self.vocab_dict)
        self.vocab_dict[self.data_tokens['pad_token']] = len(self.vocab_dict)
        # len(self.vocab_dict)
        return self.vocab_dict
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
        self.chars_to_remove_regex = '[\]\[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«\(\)]'
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
        try:
            all_text = " ".join(batch[self.transcription_field])
        except KeyError:
            print("It looks like your data may be already processed, "
            "but we need the unprocessed data to make an up-to-date "
            "tokenizer, which we want to make sure reflects this data set.")
            quit()
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
        # print(f"char_list ({len(char_list)}):", char_list)
        self.vocab_dict = {v:k for k,v in enumerate(sorted(char_list))}
        print("self.vocab_dict compiled from this data "
                f"({len(self.vocab_dict)}):",self.vocab_dict)
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
        #save to disk is after model tokenization
class Processor():
    def push_to_hub(self,repo_name):
        import pushtoken
        print(f"Pushing processor to {repo_name} repo (not yet!)")
        # self.processor.push_to_hub(repo_name,token=pushtoken.token,
        #                             hub_private_repo=self.hub_private_repo)
    def make_vocab_file(self):
        file_name=f"vocab_{self.language['iso']}.json"
        self.vocab_file_name=os.path.join(self.data_file_location,file_name)
        with open(self.vocab_file_name, 'w') as vocab_file:
            json.dump(self.vocab_dict, vocab_file)
    def verify_json(self,json_file=None):
        """If verified, language specific file is copied to filename expected
        by transformers
        """
        if not json_file:
            json_file=self.vocab_file_name
        defaults=set(self.data_tokens.values()
                            # ['<unk>','<pad>','|']
                        )
        with open(json_file, 'r') as vocab_file:
            d=json.load(vocab_file)
            defaults_present=defaults&(set(d)|set(d[self.language['iso']]))
            if len(defaults_present) == len(defaults):
                print(f"all defaults ({defaults}) found in json file.")
                with open('vocab.json', 'w') as f: #Must be in Working Directory!
                    json.dump(d,f)
            else:
                print(f"json file missing default value(s) {defaults-set(d)}")
    def make_tokenizer(self):
        """CTC tokenizers need to be built based on data, not just loaded
        from_pretrained, like other tokenizers, so we manage
        this difference here"""
        #this should just download or use from cache
        # print(f"Working with {self.processor_parent_fn.__name__} "
        #         "Processor")
        if self.tokenizer_fn.__name__ in ['Wav2Vec2CTCTokenizer']:
            """FutureWarning: Loading a tokenizer inside Wav2Vec2BertProcessor
            from a config that does not include a `tokenizer_class` attribute
            is deprecated and will be removed in v5. Please add
            `'tokenizer_class': 'Wav2Vec2CTCTokenizer'` attribute to either
            your `config.json` or `tokenizer_config.json` file to suppress
            this warning"""
            self.make_vocab_file()
            print("Building tokenizer from (local) json file.")
            try:
                self.verify_json()
            except FileNotFoundError as e:
                print(f"It looks like the json file didn't get made; "
                    "be sure to set make_vocab=True in data load ({e})")
                exit()
            loc='./' #this builds from local json
            self.processor_remade=True
            kwargs={'task':"transcribe",
                    #if this crashes BERT, will need to separate out:
                    'target_lang':self.language['iso'],
                    'tokenizer_class':self.processor_parent_fn.__name__,
                    **self.data_tokens}
        else:
            print("Downloading tokenizer or using from cache. ")
            loc=self.fqbasemodelname
            self.processor_remade=False
            kwargs={}
        self.tokenizer = self.tokenizer_fn.from_pretrained(loc,
                                            # self.fqmodelname_hf,
                                            # language=self.languagename,
                                            # task="transcribe",
                                            # cache_dir=self.cache_dir
                                            **kwargs
                                            )
        os.remove('vocab.json')
        print(f"Loaded {self.modelname} tokenizer from {loc}")
    def prepare_dataset_features(self,batch):
        audio = batch["audio"]
        try:
            # print("feature_extractor is",type(self.feature_extractor))
            batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        except Exception as e:
            # print(f"no self.feature_extractor? ({e})")
            # print("processor is",type(self.processor))
            batch["input_features"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(batch["input_features"])
        try:
            batch["labels"] = self.tokenizer(text=batch[self.transcription_field]).input_ids
        except Exception as e:
            # print(f"no self.tokenizer? ({e})")
            batch["labels"] = self.processor(text=batch[self.transcription_field]).input_ids
        return batch
    def prepare_dataset_values(self,batch):
        audio = batch["audio"]
        # batched output is "un-batched"
        batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        batch["labels"] = self.processor(text=batch["sentence"]).input_ids
        return batch
    def do_prepare_dataset(self,dataset):
        if dataset.dataset_prepared:
            print("Dataset seems to have been prepared ealier.")
            return
        #I.e., unless previously loaded, etc
        print(f"Processing Dataset {dataset.datasetprettyname}: "
            f"{dataset.dbd.column_names}, rows:{dataset.dbd.num_rows}")
        # print(dataset.dbd['train'][0]["audio"])
        if 'Wav2Vec2' in self.processor_parent_fn.__name__:
            fn=self.prepare_dataset_values
        else:
            fn=self.prepare_dataset_features
        dataset.dbd = dataset.dbd.map(fn,
            remove_columns=dataset.dbd.column_names["train"],
            # fn_kwargs={"cache_dir": "/media/kentr/hfcache/datasets"},
            num_proc=count_cpus())
        print(f"Going to save to disk as {dataset.datasetname_processed}: "
                f"{dataset.dbd.column_names}, rows:{dataset.dbd.num_rows}")
        dataset.dbd.save_to_disk(dataset.fqdatasetname_processed)
    def make_tokenizer_and_processor(self):
        # This should only happen for CTC models, which require custom tokenizers
        fq=self.fqbasemodelname #This class is only called when absent online
        if self.feature_extractor_fn.__name__ in ['Wav2Vec2FeatureExtractor']:
            print("Making a Wav2Vec2FeatureExtractor")
            kwargs={
                'feature_size':1,
                'sampling_rate':16000,
                'padding_value':0.0,
                'do_normalize':True,
                'return_attention_mask':True
            }
        else:
            kwargs={}
        print("Making a FeatureExtractor")
        self.feature_extractor = self.feature_extractor_fn.from_pretrained(fq,
                                                                    **kwargs)
        print(f"Loaded {self.modelname} feature extractor from {fq}")
        self.make_tokenizer()
        self.processor = self.processor_parent_fn(
                                feature_extractor=self.feature_extractor,
                                tokenizer=self.tokenizer)
        self.processor.save_pretrained(self.fqmodelname_loc)
        self.processor_remade=True
    def one_step_processor(self,**kwargs):
        for loc in [self.fqbasemodelname,self.fqmodelname_hf]:
            try:
                self.processor=self.processor_parent_fn.from_pretrained(loc,
                                                                    **kwargs)
                print(f"Loaded {self.modelname} processor from {loc} "
                        "in one step")
                #Keep this available for inference later:
                self.processor.save_pretrained(self.fqmodelname_loc)
                self.processor_remade=False
                return
            except:
                pass
        raise #shouldn't get here, unless both fail.
        # assert (hasattr(self.processor) and
        #         isinstance(self.processor,self.processor_parent_fn))
    def __init__(self,**kwargs):
        for k in list(kwargs):
            if debug:
                print(k,':',kwargs[k])
            if k in ['cache_dir','remake_processor']:
                setattr(self,k,kwargs.get(k))
            else:
                setattr(self,k,kwargs.pop(k))
        try:
            assert not kwargs.get('remake_processor'), 'by request'
            assert 'CTC' not in self.tokenizer_fn.__name__, 'by model type'
            self.one_step_processor(**kwargs) #succeeds or RuntimeError
            print(f"Made {self.fqbasemodelname} processor in one step")
        except (RuntimeError,AssertionError) as e:
            if 'expected str, bytes or os.PathLike object, not NoneType' in e.args:
                e=f"Probably missing a vocab file: {','.join(e.args)}"
            print(f"Remaking {self.modelname} processor ({e})")
            self.make_tokenizer_and_processor()
        print("Loaded Processor:",self.processor.__class__.__name__)
        try:
            print("Loaded Tokenizer:",self.tokenizer.__class__.__name__)
        except:
            print("Loaded Processor.Tokenizer:",
                                self.processor.tokenizer.__class__.__name__)
        try:
            print("Loaded Feature Extractor:",
                                    self.feature_extractor.__class__.__name__)
        except:
            print("Loaded Processor.FeatureExtractor:",
                            self.processor.feature_extractor.__class__.__name__)
class BaseModel():
    # def get_peftOK_layer_names(self):
    #     layer_names = []
    #     layer_types = []
    #     lora_modules=(torch.nn.Linear,
    #                     # torch.nn.Embedding,
    #                     # torch.nn.Conv2d,
    #                     # torch.nn.Conv3d,
    #                     # Conv1D,
    #                     # torch.nn.MultiheadAttention
    #                     # # torch.nn.Conv1d,
    #                 )
    #     # Recursively visit all modules and submodules
    #     for name, module in self.model.named_modules():
    #         # Check if the module is an instance of the specified layers
    #         if isinstance(module, lora_modules):
    #             # model name parsing
    #             # print('found', name, module)
    #             # print(name.split('.')[-1])
    #             layer_types.append(type(module))
    #             layer_names.append(name.split('.')[-1])
    #     print('peftOK_layer_names:',list(set(layer_names)))
    #     print('peftOK_layer_types:',list(set(layer_types)))
    #     return list(set(layer_names))
    def use_lora(self):
        # /home/kentr/bin/raspy/newASR/env/lib/python3.11/site-packages/peft/tuners/tuners_utils.py:550: UserWarning: Model with `tie_word_embeddings=True` and the tied_target_modules=['model.decoder.embed_tokens'] are part of the adapter. This can lead to complications, for example when merging the adapter or converting your model to formats other than safetensors. See for example https://github.com/huggingface/peft/issues/2018.
        self.check_for_input_embeddings_method()
        # lora_config = LoraConfig(
        #                             #per https://discuss.huggingface.co/t/unexpected-keywork-argument/91356:
        #                             # task_type=TaskType.SEQ_2_SEQ_LM,
        #                             task_type=TaskType('SEQ_2_SEQ_LM'),#'automatic-speech-recognition',
        #                             inference_mode=False,
        #                             r=8,
        #                             lora_alpha=32,
        #                             lora_dropout=0.1,
        #                             # target_modules=self.get_peftOK_layer_names(),
        #                             target_modules=["q_proj", "v_proj"],
        #                             # save_embedding_layers=True #necessary when changing vocab
        #                         )
        lora_config = LoraConfig(r=32, lora_alpha=64,
                                target_modules=["q_proj", "v_proj"],
                                lora_dropout=0.05,
                                bias="none"
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
        # The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
        # Reloading Whisper Large V3 Cer Hau Hausa_Mcv11X9 (Quantized LoRA) from source (Using `bitsandbytes` 8-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`)
        from peft import prepare_model_for_kbit_training
        """We may want to consider:
        —use_gradient_checkpointing (bool, optional, defaults to True) — If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        —gradient_checkpointing_kwargs (dict, optional, defaults to None) — Keyword arguments to pass to the gradient checkpointing function, please refer to the documentation of torch.utils.checkpoint.checkpoint for more details about the arguments that you can pass to that method. Note this is only available in the latest transformers versions (> 4.34.1).
        """
        self.model = prepare_model_for_kbit_training(
                        self.model,
                        # use_gradient_checkpointing=False,
                        gradient_checkpointing_kwargs={'use_reentrant':False})
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    def quant_config(self):
        # import intel_extension_for_pytorch as ipex
        from transformers import BitsAndBytesConfig
        """A rule of thumb is:
        use double quant if you have problems with memory,
        use NF4 for higher precision, and
        use a 16-bit dtype for faster finetuning. """
        return BitsAndBytesConfig(
                                   load_in_4bit=True,
                                   bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16
                                )
    def donothing(self):
        pass
    def check_for_input_embeddings_method(self):
        if not hasattr(self.getmodel_fn,'get_input_embeddings'):
            self.getmodel_fn.get_input_embeddings=self.donothing
    def get_model(self,**kwargs):
        print(f"Setting up get_model with kwargs {kwargs}")
        try:
            assert not getattr(self,'reload_model',False)
            print(f"Looking for saved {self.basemodelprettyname} model")
            self.model = self.getmodel_fn.from_pretrained(self.fqbasemodelname,
                **kwargs)
            print(f"Loaded {self.basemodelprettyname} model "
                    f"({self.fqbasemodelname})")
        except (Exception,AssertionError) as e:
            if isinstance(e,AssertionError):
                e="by request"
            print(f"Reloading {self.basemodelprettyname} from source ({e})")
            self.model = self.getmodel_fn.from_pretrained(self.fqbasemodelname,
                **{**kwargs,'force_download':self.reload_model})
        self.did_lora=self.did_quant=False
        if 'quantization_config' in kwargs:
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
                        'pad_token_id',
                        'load_in_8bit',
                        'device_map',
                        # 'add_adapter',
                        ]
        nonmodelkwargs=['getmodel_fn',
                        'language',
                        'lang_code',
                        'lora',
                        'quant',
                        'metric_name',
                        'modelprettyname',
                        # 'pad_token_id',
                        'basemodelprettyname',
                        'basemodelname',
                        'fqbasemodelname',
                        'push_to_hub',
                        'reload_model'
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
        for k in list(kwargs): #b/c pop
            if k in kwargstogetmodel:
                if debug:
                    print(k,kwargs[k],"(to get_model())")
                setattr(self,k,kwargs[k]) #leave in kwargs
            else:
                if debug:
                    print(k,kwargs[k],"(just for this BaseModel object)")
                setattr(self,k,kwargs.pop(k))
        if getattr(self,'quant',False): #for quant model loading
            print("Running quantization")
            kwargs={**kwargs,'quantization_config':self.quant_config()}
        self.get_model(**kwargs)
        if getattr(self,'quant',False): #for quant post-processing
            self.use_quantization()
            print('Quantization:',self.did_quant)
            if not self.did_quant:
                log.error("Quantization didn't work!")
                exit()
        else:
            print("Not using quantization")
        if getattr(self,'lora',False):
            self.use_lora()
            print('LoRA:',self.did_lora)
            if not self.did_lora:
                log.error("LoRA didn't work!")
                exit()
        else:
            print("Not using LoRA")
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
        print(f"Setting up training_arguments with kwargs {kwargs}")
        # per_device_train_batch_size:
        # pdtbs=16
        if not kwargs.get('per_device_train_batch_size'):
            kwargs['per_device_train_batch_size']=16
        # increase by 2x per 2x decrease in batch size:
        #gradient_accumulation_steps:
        pdtbs=kwargs.get('per_device_train_batch_size')#can be >16, e.g., 32
        gas=max(16,pdtbs)//kwargs.get('per_device_train_batch_size')
        if self.lora:
            lora_kwargs={
                # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
                'remove_unused_columns':False,
                'label_names':["labels"],  # same reason as above
                }
        else:
            lora_kwargs={}
        # What to do with remove_unused_columns?
        return self.training_args_fn(
                            **lora_kwargs,
                            # change to a repo name of your choice:
                            output_dir=self.fqmodelname_loc,
                            # per_device_train_batch_size=pdtbs,
                            gradient_accumulation_steps=gas,
                            # warmup_steps=500,
                            group_by_length=True,
                            # Not for LoRA:
                            # gradient_checkpointing=True,
                            fp16=True,
                            # eval_strategy="steps",
                            # evaluation_strategy
                            report_to=["tensorboard"],
                            load_best_model_at_end=True,
                            #pick from 'eval_loss', 'eval_errorrate', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch'
                            metric_for_best_model=self.metric_name,
                            greater_is_better=False,
                            # remove_unused_columns=False,
                            **kwargs)
    def train(self):
        """
        /home/kentr/bin/raspy/newASR/env/lib/python3.11/site-packages/torch/utils/data/dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
        warnings.warn(warn_msg)

        Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.

        `use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`...
        """
        print(f"Saving trained model to {self.fqmodelname_loc}")
        try:
            self.trainer.train(**self.train_kwargs)
        except ValueError as e:
            if 'No valid checkpoint found in output directory' in e.args[0]:
                del self.train_kwargs['resume_from_checkpoint']
                self.trainer.train(**self.train_kwargs)
            else:
                print(f"Other ValueError: ({e.args}; {e.add_note()})")
        except Exception as e:
            print(f"unknown exception: ({e})")
            raise
        if getattr(self.names,'train_adaptor_only',False):
            self.save_adaptor()
    def save_adaptor(self):
        from safetensors.torch import save_file as safe_save_file
        from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
        import os
        adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(self.language['iso'])
        self.fqmodelname_loc
        adapter_file = os.path.join(self.trainer.args.output_dir, adapter_file)
        safe_save_file(self.model._get_adapters(),
                        adapter_file,
                        metadata={"format": "pt"})
    def push(self):
        self.trainer.push_to_hub(**self.push_kwargs())
        self.processor.push_to_hub(self.modelname, **self.push_kwargs())
        # self.tokenizer.push_to_hub(self.modelname, **self.push_kwargs())
    def push_kwargs(self):
        kwargs= {
                    "dataset_tags": self.fqdatasetname,
                    # a 'pretty' name for the training dataset
                    "dataset": self.datasetprettyname,
                    "dataset_args": f"config: {self.language['mcv_code']}, split: test",
                    "language": self.language['iso'],
                    # a 'pretty' name for your model
                    "model_name": f"{self.modelname} "
                                f"{self.language['iso']} - {self.language['name']}",
                    "finetuned_from": self.fqbasemodel,
                    "tasks": "automatic-speech-recognition",
                }
        try:
            import pushtoken
            kwargs.update({"token": pushtoken.token})
        except:
            pass
        return kwargs
    def demo(self):
        d=Demo(self)
    def __init__(self,**kwargs):
        not_for_training_arguments_kwargs=[#these are for this module
                                            'fqbasemodelname',
                                            'datasetprettyname',
                                            'fqdatasetname',
                                            'fqmodelname_loc',
                                            'model',
                                            'processor',
                                            'data',
                                            'data_collator',
                                            'compute_metrics',
                                            'metric_name',
                                            'modelname',
                                            'data_collator_fn',
                                            'training_args_fn',
                                            'trainer_fn',
                                            'compute_metrics_fn_name',
                                            'lora',
                                            'language',
                                        ]
        for_trainer_kwargs=[#args to .train(), just what huggingface wants
            'resume_from_checkpoint'
        ]
        for_training_arguments_kwargs=[#args to .train(args=), for huggingface
            'eval_strategy',
            'save_strategy',
            'per_device_train_batch_size',
            'warmup_steps',
            'learning_rate',
            'predict_with_generate',
            'gradient_checkpointing',
            'save_steps',
            'eval_steps',
            'logging_steps',
            'save_total_limit',
            'num_train_epochs',
            'lr_scheduler_type',
            'dataloader_pin_memory',
            'push_to_hub',
            'hub_private_repo',
            ]
        overlap=(set(for_training_arguments_kwargs)&set(
                                            not_for_training_arguments_kwargs)
                ) or (
                    set(for_trainer_kwargs)&set(
                                            not_for_training_arguments_kwargs)
                ) or set(for_training_arguments_kwargs)&set(for_trainer_kwargs)
        remaining=set(kwargs)-set(for_training_arguments_kwargs)-set(
                    not_for_training_arguments_kwargs)-set(for_trainer_kwargs)
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
        self.train_kwargs={}
        for k in kwargs.copy(): #b/c pop
            if k in for_training_arguments_kwargs:
                if debug:
                    print(k,kwargs[k],"(to training_arguments())")
                setattr(self,k,kwargs[k]) #leave in kwargs
            elif k in for_trainer_kwargs:
                if debug:
                    print(k,kwargs[k],"(to train())")
                self.train_kwargs[k]=kwargs.pop(k)
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
                            #Can't tell if either of the following matters:
                            # tokenizer=self.processor.feature_extractor #not tokenizer
                            processing_class=self.processor.feature_extractor, #not tokenizer
                        )
class TrainWrapper(object):
    def get_data(self,data_tokens):
        self.data=Data(data_tokens=data_tokens, **self.names.datakwargs())
    def get_processor(self,data_tokens):
        """Do I need this Processor wrapper? will I ever need a second?
        Hardcoding this here right now seems best, at least for now"""
        kwargs=self.names.processorkwargs()
        if self.names.tokenizer_fn.__name__ in ['Wav2Vec2CTCTokenizer']:
            tokenizer=self.names.tokenizer_fn.from_pretrained(
                                                    self.names.fqbasemodelname)
            vocab_dict=tokenizer.vocab
            vocab_dict.update({self.names.language[
                                            'iso']:self.data.make_vocab_dict()})
            kwargs={**kwargs,'data_tokens':data_tokens,
                            'vocab_dict':vocab_dict}
        processor=Processor(**kwargs)
        if processor.processor_remade:
            #This is important because we want data preprocessed by the correct
            # processor. So we do that now if it wasn't already done, or if
            # it was done by an earlier processor.
            self.data.dataset_prepared=False
        if not self.data.dataset_prepared:
            try:
                self.data.show_dimensions_preprocessed()
            except KeyError: #the data as we have it is already processed
                print("Reloading data which looks already processed by an "
                    "earlier processor")
                self.get_data()
            processor.do_prepare_dataset(self.data)
            self.data.cleanup() #data temp files
        else:
            print("Data looks already prepared")
        if getattr(self.names,'push_to_hub',False) and self.names.fqmodelname_hf:
            print(f"Going to push to {self.names.fqmodelname_hf} repo (not yet)")
            # processor.push_to_hub(self.names.fqmodelname_hf)
        if hasattr(processor,'tokenizer'):
            self.tokenizer=processor.tokenizer
        elif hasattr(processor.processor,'tokenizer'):
            self.tokenizer=processor.processor.tokenizer
        if hasattr(processor,'feature_extractor'):
            self.feature_extractor=processor.feature_extractor
        elif hasattr(processor.processor,'feature_extractor'):
            self.feature_extractor=processor.processor.feature_extractor
        #Processor object goes away at this point
        self.processor=processor.processor
    def get_base_model(self):
        kwargs=self.names.modelkwargs()
        if self.names.processor_parent_fn.__name__ in ['Wav2Vec2ForCTC',
                                                        'Wav2Vec2BertForCTC']:
            kwargs.update({
                        'vocab_size':len(self.processor.tokenizer),
                        'pad_token_id':self.processor.tokenizer.pad_token_id,
                        })
        model=BaseModel(
                        **kwargs
                        )
        #BaseModel object goes away at this point
        self.model=model.model
        if self.names.processor_parent_fn.__name__ in [
                                            'WhisperForConditionalGeneration']:
                self.set_whisper_langs()
    def set_whisper_langs(self):
        # print("possible language codes: "
        # f"{[i.strip('<>|') for i in self.model.generation_config.lang_to_id.keys()]}")
        self.model.generation_config.language = self.names.sister_language['mcv_code']
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        # self.model.generation_config.use_cache=True
        self.model.generation_config.use_cache=self.names.use_cache_in_training
    def train(self):
        self.trainer.train()
        self.model.save_pretrained(self.names.fqmodelname_loc)
    def push(self):
        self.trainer.push()
    def infer(self):
        import infer
        fqmodelnames_loc=[self.names.fqmodelname_loc]
        models=infer.InferDict(fqmodelnames_loc,checkpoints='infer_checkpoints')
        print(f"Working with audio to infer: {self.names.audio}")
        if not self.names.audio:
            if self.names.language['iso'] == 'gnd': #set a few defaults for test languages
                print("setting default audio to infer for Zulgo")
                self.names.audio=[
                    '/home/kentr/Assignment/Tools/WeSay/gnd/ASR/'
                    'Listen_to_Bible_Audio_-_Mata_2_-_Bible__Zulgo___gnd___'
                    'Audio_Bibles-MAT_2_min1.wav'
                    ]
        for file in self.names.audio:
            show_standard=True #just once per audio
            for m in models:
                print(os.path.basename(m)+':', models[m](file,show_standard))
                show_standard=False #just once each time
    def notify_user_todo(self):
        t=[m for m in ['train','demo','infer'] if getattr(self.names,m,False)]
        if len(t) > 2:
            todo=[', '.join(t[:-1]),t[-1]] #just the last
        if len(t) > 1:
            t.insert(-1,'and')
        t=' '.join(t)
        print(f"going to {t if t else 'nothing?!?'}")
    def get_names(self,model_type,trainer_type,my_options_args):
        # print(f"use_cache_in_training: {my_options_args['use_cache_in_training']}")
        kwargs={**model_type}
        kwargs.update(trainer_type)
        kwargs.update(**my_options_args) #pull in default and user settings
        self.names=Nomenclature(**kwargs)
        # print(f"use_cache_in_training: {self.names.use_cache_in_training}")
        # exit()
        if not isinstance(self.names,Nomenclature):
            print(f"Found ({type(names)}) names object; errors may follow.")
            self.names=Nomenclature()
    def init_debug(self):
        self.notify_user_todo()
        if getattr(self.names,'debug',False):
            for attr in dir(self.names):
                if '__' not in attr:
                    print(attr,getattr(self.names,attr))
    def get_data_processor_model(self):
        if (getattr(self.names,'train',False) or
            getattr(self.names,'push_to_hub',False)):
            data_tokens={
                        'unk_token':"<unk>",
                        'pad_token':"<pad>",
                        'word_delimiter_token':"|",
                        }
            self.get_data(data_tokens)
            self.get_processor(data_tokens)
            self.get_base_model()
    def compute_metrics_whisper(self,pred):
        # print("Running train_whisper.compute_metrics")
        # print(f"Running with data {pred} ({type(pred)})")
        # print(f"Running with pred.predictions {pred.predictions} ({type(pred.predictions)})")
        # print(f"Running with pred.label_ids {pred.label_ids} ({type(pred.label_ids)})")
        # print(f"Running with self.tokenizer {self.tokenizer} ({type(self.tokenizer)})")
        # print(f"Running with self.tokenizer.batch_decode {self.tokenizer.batch_decode} ({type(self.tokenizer.batch_decode)})")
        # print(f"Running with self.tokenizer.pad_token_id {self.tokenizer.pad_token_id} ({type(self.tokenizer.pad_token_id)})")
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # with open('metrics_log.txt','a') as f:
        #     f.write(f"inferred:{pred_str}\n")
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # with open('metrics_log.txt','a') as f:
        #     f.write(f"should have inferred:{label_str}\n")
        # with open('metrics_log.txt','a') as f:
        #     f.write(f"inferred: {pred_str}; correct:{label_str}")

        error = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        with open('metrics_log.txt','a') as f:
            for n in range(len(pred_str)):
                f.write(f"inferred:{pred_str[n]}\n")
                f.write(f"correct:{label_str[n]}\n")
            f.write(f"{self.names.metric_name}: {error}\n")
        return {self.names.metric_name: error}
    def compute_metrics_ctc(self,pred):
        """This needs to be here because the processor may be loaded
        without the train class around it."""
        pred_logits = pred.predictions
        pred_ids = numpy.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        error = self.metric.compute(predictions=pred_str, references=label_str)

        return {self.names.metric_name: error}
    def get_data_collator(self):
        if self.names.processor_parent_fn.__name__ in ['Wav2Vec2BertProcessor',
                                                        'Wav2Vec2Processor']:
            kwargs={'processor':self.processor,'padding':True}
        elif self.names.processor_parent_fn.__name__ in ['WhisperProcessor']:
            kwargs={'processor':self.processor,
                    'decoder_start_token_id':
                                    self.model.config.decoder_start_token_id}
        else:
            print("I don't have a data collator set up for "
                    f"{self.processor_parent_fn.__name__} processor!")
            exit()
        self.data_collator = self.names.data_collator_fn(**kwargs)
    def freeze_all_but_adaptor_layers(self):
        self.model.init_adapter_layers()
        self.model.freeze_base_model()
        adapter_weights = self.model._get_adapters()
        for param in adapter_weights.values():
            param.requires_grad = True
        print("Froze all but adaptor layers")
    def get_trainer(self):
        if (getattr(self.names,'train',False) or
            getattr(self.names,'push_to_hub',False)):
            self.get_data_collator()
            compute_metrics=getattr(self, self.names.compute_metrics_fn_name)
            self.trainer=Training(
                        model=self.model,
                        processor=self.processor, #downloads or builds above
                        data=self.data.dbd,
                        data_collator=self.data_collator,
                        compute_metrics=compute_metrics,
                        **self.names.trainingkwargs()
                        )
    def do_stuff(self):
        if getattr(self.names,'train',False):
            self.trainer.train()
        if getattr(self.names,'push_to_hub',False): #token import in train.py
            self.trainer.push()
        if getattr(self.names,'demo',False):
            if not hasattr(self,'processor'):
                self.get_processor()
            if self.names.fqmodelname_loc:
                Demo(self.names)
        if getattr(self.names,'infer',False):
            self.infer()
    def load_and_do_stuff(self):
        self.get_data_processor_model()
        if getattr(self.names,'train_adaptor_only',False):
            self.freeze_all_but_adaptor_layers()
        self.get_trainer()
        #in compute_metrics only:
        self.metric = evaluate.load(self.names.metric_name)
        self.do_stuff()
    def __init__(self,model_type,trainer_type,my_options,do_later=False):
        my_options.sanitize() # wait until everyting is set to do this
        self.get_names(model_type,trainer_type,my_options.args)
        self.init_debug()
        if not do_later:
            self.load_and_do_stuff()
class Demo(object):
    def transcribe_module(self,audio):
        return self.inferer(audio,show_standard=True)
    def transcribe_pipe(self,audio):
        return self.pipe(audio)["text"]
    def do_pipe(self):
        from transformers import pipeline
        self.pipe = pipeline(model=names.fqmodelname_loc,
                            model_kwargs={"cache_dir": names.cache_dir},
                            tokenizer=names.fqmodelname_loc,
                            task='automatic-speech-recognition',
                            )
        inputs1=gr.Audio(sources=['microphone', 'upload'], type="filepath"),
        iface_pipe = gr.Interface(
            fn=self.transcribe_pipe,
            inputs=inputs1,
            outputs="text",
            title=f"Automatic Speech Recognition (ASR) {names.language['name']}",
            description=(f"Realtime demo for {names.language['name']} speech "
                        "recognition using a fine-tuned "
                        f"{names.modelprettyname} model."),
            )
        iface_pipe.launch()
    def do_tabs(self):
        app = gr.TabbedInterface(interface_list=[iface_pipe, iface_module],
                         tab_names = ["pipe", "module"])
        app.launch()
    def __init__(self,names):
        import gradio as gr
        import infer
        self.inferer=infer.Infer(names.fqmodelname_loc)
        if not self.inferer.loaded:
            print(f"Not loading demo; {names.fqmodelname_loc} model "
                "didn't load (is it trained?)")
            return
        inputs2=gr.Audio(sources=['microphone', 'upload'], type="filepath"),
        iface_module = gr.Interface(
            fn=self.transcribe_module,
            inputs=inputs2,
            outputs="text",
            title=f"Automatic Speech Recognition (ASR) {names.language['name']}",
            description=(f"Realtime demo for {names.language['name']} speech "
                        "recognition using a fine-tuned "
                        f"{names.modelprettyname} model."),
            )
        iface_module.launch()
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
            'processor_parent_fn',
            # 'tokenizer_fn_kwargs',
            'transcription_field',
            'metric',
            'metric_name',
            'fqmodelname_hf',
            'fqmodelname_loc',
            'fqbasemodelname',
            'modelname',
            'cache_dir',
            'language',
            'sister_language',
            'data_file_location',
        ]
        return {a:getattr(self,a) for a in attrs if hasattr(self,a)}
    def datakwargs(self):
        attrs=['mcv_code',
                'dataset_code',
                'data_files',
                'data_file_location',
                'language',
                'sister_language',
                'dataset_dir',
                'transcription_field',
                'max_data_rows',
                'refresh_data',
                'no_capital_letters',
                'no_special_characters',
                # 'make_vocab',
                'fqdatasetname',
                'datasetprettyname',
                'data_splits',
                'datasetname_processed',
                'fqdatasetname_processed',
                'proportion_to_train_with',
                'hub_private_repo',
                'processor_parent_fn',
            ]
        return {a:getattr(self,a) for a in attrs if hasattr(self,a)}
    def modelkwargs(self):
        attrs=['fqbasemodelname',
                'modelprettyname',
                'basemodelprettyname',
                'getmodel_fn',
                'cache_dir',
                'reload_model',
                'layerdrop',
                'attention_dropout',
                'hidden_dropout',
                'feat_proj_dropout',
                'mask_time_prob',
                'ctc_loss_reduction',
                'language',
                'iteration',
                'lora',
                'quant',
                'load_in_8bit',
                'device_map',
            ]
        return {a:getattr(self,a) for a in attrs if hasattr(self,a)}
    def trainingkwargs(self):
        attrs=['fqbasemodelname',
                'fqmodelname_loc',
                'datasetprettyname',
                'fqdatasetname',
                'resume_from_checkpoint',
                'data_collator_fn',
                'training_args_fn',
                'trainer_fn',
                'data_collator_fn',
                'compute_metrics_fn_name',
                'training_args_fn',
                'trainer_fn',
                'predict_with_generate',
                'per_device_train_batch_size',
                'dataloader_pin_memory',
                'eval_strategy',
                'save_strategy',
                'compute_metrics',
                'metric_name',
                'learning_rate',
                'lr_scheduler_type',
                'save_steps',
                'eval_steps',
                'logging_steps',
                'save_total_limit',
                'num_train_epochs',
                'lora',
                'language',
                'push_to_hub'
            ]
        return {a:getattr(self,a) for a in attrs if hasattr(self,a)}
    def init_languages(self):
        self.languages={'gnd':{'mcv_code':'gnd', 'iso':'gnd', 'name':'Zulgo'},
                        "yo":{'mcv_code':'yo', 'iso':'yor', 'name':"Yoruba"},
                        "sw":{'mcv_code':'sw', 'iso':'swh', 'name':"Swahili"},
                        "ig":{'mcv_code':'ig', 'iso':'ibo', 'name':"Igbo"},
                        "rw":{'mcv_code':'rw', 'iso':'kin', 'name':"Kinyarwanda"},
                        "lg":{'mcv_code':'lg', 'iso':'lug', 'name':"Luganda"},
                        "chr":{'iso':'chr', 'name':"Cherokee"},
                        "hau":{'mcv_code':'ha','iso':'hau','name':"Hausa"}
                        }
    def setlang(self,**kwargs):
        self.init_languages()
        language_iso=kwargs.get('language_iso')
        sister_language_iso=kwargs.get('sister_language_iso')
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
        if sister_language_iso:
            self.sister_language=self.languages[sister_language_iso]
        else: #We may update a model with new data before finetuning to a sister
            self.sister_language=self.languages[language_iso]
    def noparens_dirs(self,x):
        return '_'.join(x.translate(self.parens_dirs_dict).split(' '))
    def dataset_name(self,**kwargs):
        if (hasattr(self,'data_file_prefixes') and
                getattr(self,'data_file_prefixes') and not self.dataset_code):
            self.dataset_code='csv'
        self.parens_dirs_dict=str.maketrans('','',')(')
        d={
            'mcv11':"mozilla-foundation/common_voice_11_0",
            'mcv17':"mozilla-foundation/common_voice_17_0",
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
        # self.dataset_configbits+=['to_train']+self.fqbasemodelname.split('/')
        # The exact model type/size (above) isn't as relevant to processed data
        # as the processor used (below)
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
        if getattr(self,'quant',False):
            tuned_bits+=['quant']
            prettybits+=['Quantized']
        if getattr(self,'lora',False):
            tuned_bits+=['lora']
            prettybits+=['LoRA']
        # tuned_bits+=[self.metric_name] #This shouldn't be relevant
        tuned_bits+=[self.language['iso'],self.language['name']]
        if kwargs.get('increment'):
            tuned_bits+=[str(self.increment)] #in case this is int
        self.modelname='-'.join(tuned_bits)
        if (self.dataset_code == 'csv' and
                getattr(self,'data_file_prefixes',None) and
                self.data_file_prefixes[0].split('_')[-1].isdigit()):
            data_source=str(sum([int(i.split('_')[-1]) for i in self.data_file_prefixes]))
            # print(f"using {rows} rows")
        elif self.dataset_code == 'csv':
            print("getattr data_file_prefixes:",getattr(self,'data_file_prefixes',None))
            print("int:",int(self.data_file_prefixes[0].split('_')[-1]))
            print("int:",self.data_file_prefixes[0].split('_')[-1].isdigit())
            print("int options:",self.data_file_prefixes)
            data_source=''
        else:
            data_source=self.dataset_code
        self.modelname+=f'_{data_source}'#x{str(self.num_train_epochs)}'
        cache=self.cache_dir_tuned if self.cache_dir_tuned else self.cache_dir
        self.fqmodelname_loc=os.path.join(
                                        cache,
                                        self.modelname)
        if getattr(self,'my_hf_login',False):
            self.fqmodelname_hf='/'.join([self.my_hf_login,self.modelname])
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
    def sanitize(self):
        if 'CTC' in self.tokenizer_fn.__name__:
            self.refresh_data=True
        if not torch.cuda.is_available():
            self.dataloader_pin_memory=False
            if self.quant:
                print("Asked to quantize model, but no GPU present, so skipping.")
                self.quant=False
    def __init__(self,**kwargs):
        self.setlang(**kwargs)
        for k in kwargs:
            if debug:
                print(k,kwargs[k])
            setattr(self,k,kwargs[k])
        # if kwargs.get('fqmodelname'):
        #     self.modelname=self.fqmodelname.split('/')[-1] #just the name
        self.sanitize()
        if kwargs.get('fqbasemodelname'):
            self.basemodelname=self.fqbasemodelname.split('/')[-1]
        else:
            print("You really need to specify your full model address/name. "
                    "For inference, this is used to find the name of the "
                    "model to use for inference.")
            raise
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
def count_cpus():
    #Not sure if this is working; override:
    # self.cpus=10
    # return
    import multiprocessing #not working??!?
    return multiprocessing.cpu_count()

if __name__ == '__main__':
    import sys,os
    import train #local
    if not train.in_venv():
        print("this script is mean to run in a virtual environment, but isn't.")
    my_options=train.options.Parser('train','infer')
    # my_options=make_options()
    """‘refresh_data’ happens anyway if processor is remade"""
    """‘remake_processor’ happens anyway if not found"""
    """‘reload_model’ causes a large download!"""

    """Pick one:"""
    from model_configs import whisper
    model_type=whisper()
    model_type.update({'fqbasemodelname':
                    "kent-rasmussen/whisper-large-v3-cer-hau-Hausa_mcv11x9"})

    # from model_configs import mms
    # model_type=mms()
    # model_type.update({'fqbasemodelname':"facebook/mms-1b-all"})

    print(f"working with model {model_type['fqbasemodelname']}")
    """This is general settings"""
    my_options.args.update({
                            'cache_dir':'/media/kentr/hfcache',
                            'data_file_location':'training_data',
                            'train':True,
                            # 'infer':True,
                            # 'infer_checkpoints':True,
                            # 'refresh_data':True, #any case w/new processor or CTC
                            # 'remake_processor':True, #any case if not found
                            # 'reload_model':True #large download!
                            'lora':True,
                            # 'quant':True, #not for cpu
                            'debug':True
                        })
    """Pick one of the following four:"""
    # my_options.args.update(train.options.minimal_zulgo_test_options())
    my_options.args.update(train.options.maximal_zulgo_test_options())
    # my_options.args.update({
    #                         'dataset_code':'mcv17',
    #                         'data_splits':['train','validation'],
    #                         'language_iso':'hau',
    #                         'sister_language_iso': 'hau',
    #                         # 'max_data_rows':20,
    #                         })
    # my_options.args.update({
    #                         'dataset_code':'csv',
    #                         'language_iso':'gnd',
    #                         'data_file_prefixes':['lexicon_640'],
    #                         # 'data_file_prefixes':['examples_4589'],
    #                         # 'data_file_prefixes':['lexicon_13'],
    #                         })
    my_options.sanitize()
    # print(my_options.args)
    trainer_type={
                # **train.options.quantization(),
                'gradient_checkpointing':True,
                # 'learning_rate':1e-5,
                'learning_rate':1e-3,
                'load_best_model_at_end':True,
                # 'per_device_train_batch_size':8,
                # 'per_device_train_batch_size':16,
                # 'per_device_train_batch_size':32,
                'per_device_train_batch_size':64,
                # 'per_device_train_batch_size':128,
                # load_best_model_at_end requires the save and eval strategy to match
                # 'eval_strategy': 'epoch',
                # 'save_strategy': 'epoch',
                'eval_strategy': 'steps',
                'save_strategy': 'steps',
                'save_steps':5,
                'eval_steps':5,
                # 'logging_steps':20,
                'save_total_limit':3,
                'num_train_epochs':3,
                # 'attention-dropout':0.0,
                # 'hidden-dropout':0.0,
                # 'feat-proj-dropout':0.0,
                # 'mask-time-prob':0.0,
                # 'layerdrop':0.0,
                # 'ctc-loss-reduction':'mean'
                'resume_from_checkpoint':True,
                # above if last checkpoint is saved correctly, below if not
                # 'resume_from_checkpoint':os.path.join(
                #                 my_options.args['cache_dir'],
                #                 "mms-1b-all-cer-gnd-Zulgo_640x6/checkpoint-18")
            }
    my_options.args['metric_name']='wer'
    # for my_options.args['metric_name'] in ['cer','wer']:
    tw=train.TrainWrapper(model_type,trainer_type,my_options)
    tw.load_and_do_stuff()
    exit()
    for my_options.args['data_file_prefixes'] in [
                                        ['lexicon_640','examples_300'],
                                    ]:
        TrainWrapper(model_type,trainer_type,my_options.args)
