#!/usr/bin/env python3
# coding=UTF-8
# This was originally based on https://huggingface.co/blog/fine-tune-w2v2-bert,
# though it has been generalized and adapted from other sources, as well.

# from transformers import TrainingArguments, Trainer, AutoModelForCTC
# from transformers import Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor
# from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
from transformers import pipeline
from transformers import BitsAndBytesConfig
from dataclasses import dataclass
import evaluate
import torch
import torchaudio
import numpy
from typing import Any, Dict, List, Union
import sys,os
import datetime
import json
import train #local

def marktime(f,*args,**kwargs):
    def timed(*args,**kwargs):
        start_time=datetime.datetime.now(datetime.UTC)
        r=f(*args,**kwargs)
        print("Function",f.__class__.__name__,datetime.datetime.now(datetime.UTC)-start_time)
        return r
    return timed
class Data(train.Data):
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
    def __init__(self,**kwargs):
        kwargs['make_vocab']=True
        super().__init__(**kwargs)
class TrainWrapper(train.TrainWrapper):
    def __init__(self,model_type,trainer_type,my_options_args):
        self.get_names(model_type,trainer_type,my_options_args)
        self.init_debug()
        self.processor_fn_kwargs=self.names.processorkwargs()
        self.get_data_processor_model()
        self.get_trainer()
        super().__init__()
        #in compute_metrics only:
        self.metric = evaluate.load(self.names.metric_name)
        self.do_stuff()
def make_options():
    import options
    return options.Parser('train','infer')
if __name__ == '__main__':
    if not train.in_venv():
        print("this script is mean to run in a virtual environment, but isn't.")
    my_options=make_options()
    """‘refresh_data’ happens anyway if processor is remade"""
    """‘remake_processor’ happens anyway if not found"""
    """‘reload_model’ causes a large download!"""
    my_options.args.update({
            'language_iso':'gnd',
            'cache_dir':'/media/kentr/hfcache',
            'dataset_code':'csv',
            'data_file_prefixes':['lexicon_640'],#,'examples_4589'
            'data_file_location':'training_data',
            'infer_checkpoints':True,
            'train':True,
            'infer':True,
            # 'refresh_data':True,
            # 'remake_processor':True, #
            # 'reload_model':True
        })
    if 'google.colab' in sys.modules:
        print("It looks like we're running in a colab instance, so setting "
            "some variables now.")
        my_options.args.update({
                'cache_dir':'.',
            })
    my_options.sanitize() # wait until everyting is set to do this
    from model_configs import w2v_bert_2
    model_type=w2v_bert_2()
    # {
    #             'fqbasemodelname':"facebook/w2v-bert-2.0",
    #             'getmodel_fn':Wav2Vec2BertForCTC, #for tuned models
    #             'tokenizer_fn':Wav2Vec2CTCTokenizer,
    #             'feature_extractor_fn':SeamlessM4TFeatureExtractor,
    #             # 'processor_fn':Wav2Vec2BertProcessor,
    #             'processor_fn':Processor,
    #             }
    trainer_type={
                # 'data_collator_fn':DataCollatorCTCWithPadding,
                # 'training_args_fn':TrainingArguments,
                # 'trainer_fn':Trainer,
                'learning_rate':1e-5,
                'per_device_train_batch_size':16,
                'save_steps':20,
                # load_best_model_at_end requires the save and eval strategy to match
                'eval_strategy': 'epoch',
                'save_strategy': 'epoch',
                # 'eval_steps':5,
                'logging_steps':20,
                'save_total_limit':6,
                'num_train_epochs':1,
                'attention-dropout':0.0,
                'hidden-dropout':0.0,
                'feat-proj-dropout':0.0,
                'mask-time-prob':0.0,
                'layerdrop':0.0,
                'ctc-loss-reduction':'mean'
                # 'compute_metrics' is hardcoded; add flexibility if needed
                }
    for my_options.args['metric_name'] in ['cer','wer']:
        TrainWrapper(model_type,trainer_type,my_options.args)
    exit()
    for my_options.args['data_file_prefixes'] in [
                                        ['lexicon_640','examples_300'],
                                    ]:
        TrainWrapper(model_type,trainer_type,my_options.args)
