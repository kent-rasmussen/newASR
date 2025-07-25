#!/usr/bin/env python3
# coding=UTF-8
import sys
import os
from datasets import Dataset, Audio #needed for Infer()
time_model_load=False
time_inference=True
# from transformers import TrainingArguments, Trainer, AutoModelForCTC
# from transformers import Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor
# from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor,
from transformers import AutoProcessor, AutoModel
# from transformers import pipeline
# from transformers import BitsAndBytesConfig
# from dataclasses import dataclass
# import evaluate
import torch
import torchaudio
from typing import Any, Dict, List, Union
import datetime

def marktime(do=True):
    def wrapper(f):
        def keeptime(*args,**kwargs):
            start_time=datetime.datetime.now(datetime.UTC)
            r=f(*args,**kwargs)
            print("Function",f.__qualname__,
                    datetime.datetime.now(datetime.UTC)-start_time)
            return r
        if do:
            return keeptime
        else:
            return f
    return wrapper
class Infer(object):
    @marktime(do=time_inference)
    def __call__(self,x,show_standard=False):
        # print(f"Inferring {x}")
        if show_standard:
            self.standard(x)
        audio_dataset = Dataset.from_dict({"audio": [x]}).cast_column("audio",
                                            Audio(sampling_rate=16000))
        sampling_rate = audio_dataset.features["audio"].sampling_rate
        inputs = self.processor(audio_dataset[0]["audio"]["array"],
                                sampling_rate=sampling_rate,
                                return_tensors="pt")
        with torch.no_grad(): #for inference
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        r=self.processor.batch_decode(predicted_ids)
        return r
    def standard(self,file):
        if os.path.isfile(os.path.splitext(file)[0]+'.txt'):
            with open(os.path.splitext(file)[0]+'.txt') as f:
                print("Good Transcription:",f.read())
    def getprocessor(self):
        proc_fn=AutoProcessor
        for repo in self.repos:
            # print(f"Trying to load processor at {repo}.")
            try:
                self.processor = proc_fn.from_pretrained(repo)
                # AutoProcessor
                # print(f"Processor at {repo} loaded.")
                break
            except (OSError,TypeError): #i.e., not on local filesystem
                # print(f"Trying to load checkpoint processor above {repo}.")
                totry=os.path.dirname(repo)
                # print(f"Trying to load checkpoint processor at {totry}.")
                self.processor = proc_fn.from_pretrained(totry)
                # print("Processor loaded from checkpoint parent")
                break
            except Exception as e:
                print(f"Other exception: ({e})")
                raise
        if hasattr(self,'processor') and isinstance(self.processor,proc_fn):
            return
            # print(f"Processor: {type(self.processor)}")
        else:
            print("Processor not on local filesystem")
    def get_model_fn(self,repo):
        self.kwargs={}
        if 'mms' in repo:
            from transformers import Wav2Vec2ForCTC
            self.kwargs={'ignore_mismatched_sizes':True}
            return Wav2Vec2ForCTC
        if 'w2v' in repo:
            from transformers import Wav2Vec2BertForCTC
            return Wav2Vec2BertForCTC
        if 'whisper' in repo:
            from transformers import AutoModelForSpeechSeq2Seq
            return AutoModelForSpeechSeq2Seq
        return AutoModel
    @marktime(do=time_model_load)
    def __init__(self,*repos):
        # if torch.cuda.is_available()
        self.repos=repos
        self.loaded=False
        # print(f"trying these repos: {str(repos).strip('(),')}")
        self.getprocessor()
        for repo in repos:
            # print(f"Trying to load model at {repo}.")
            try:
                self.model = self.get_model_fn(repo).from_pretrained(repo,
                            vocab_size=len(self.processor.tokenizer),
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            **self.kwargs
                        )
                print(f"Model at {repo} loaded.")
                self.loaded=True
                break
            except OSError: #i.e., not on local filesystem
                print(f"Model not found at {repo}.")
                pass
            except ValueError as e:
                print("there may be a configuration problem with this model "
                    f"({e})")
        if not hasattr(self,'model'):
            print(f"sorry, model(s) at "
                    f"{str(repos).strip('(),')} not loaded; "
                    "inference will fail.")
        elif not isinstance(self.model, self.get_model_fn(repo)):
            print(f"sorry, {type(self.model).__name__} model(s) at "
                    f"{str(repos).strip('(),')} not loaded; "
                    "inference will fail.")
class InferDict(dict):
    def infer_checkpoints(self):
        for m in self.models.copy():
            checkpoints=[i.path for i in os.scandir(m)
                        if 'checkpoint-' in i.name
                        if os.path.isdir(i)
                        ]
            # print("found checkpoints",checkpoints)
            for c in checkpoints:
                self.models.insert(self.models.index(m),c)
    def __init__(self,models,checkpoints=False):
        self.models=models
        if checkpoints:
            self.infer_checkpoints()
        print("Trying to load models at",self.models)
        for i in self.models:
            self[i]=Infer(i)
            if not self[i].loaded:
                del self[i]
        print("going to infer",self.models)
if __name__ == '__main__':
    import options
    model_cache='/media/kentr/Backups/hfcache/'
    files=[
        '/home/kentr/Assignment/Tools/WeSay/gnd/ASR/'
        'Listen_to_Bible_Audio_-_Mata_2_-_Bible__Zulgo___gnd___'
        'Audio_Bibles-MAT_2_min1.wav',
        # '/home/kentr/Assignment/Tools/WeSay/gnd/ASR/'
        # 'Listen_to_Bible_Audio_-_Mata_2_-_Bible__Zulgo___gnd___'
        # 'Audio_Bibles-MAT_2_1of4.wav'
    ]
    analang='gnd'
    do_checkpoints=True #check inference for each saved checkpoint.
    modellist=[ #build list programmatically (or list below)
        d.path for d in os.scandir(model_cache)
        if f'-{analang}-' in d.name
        if '-lora-' in d.name
        if '2140x3' in d.name
        if '-wer-' in d.name
        # if '-cer-' in d.name
        if d.is_dir() #others should exclude faster
    ]
    if do_checkpoints:
        modellist=infer_checkpoints(modellist)
    # print('\n'.join(modellist))

    models={} #because this loads multiple models first, store in dict.
    if len(sys.argv)>2:
        modellist=sys.argv[1:2]
        files=sys.argv[2:]
    elif sys.argv[1:]:
        files=sys.argv[1:]
        # if hasattr(models[m],'model'):
        #     print(f"{model} loaded")
    for file in files:
        titled=False
        print(f"Inferring {file}")
        # if os.path.isfile(os.path.splitext(file)[0]+'.txt'):
        #     with open(os.path.splitext(file)[0]+'.txt') as f:
        #         print("Good Transcription:",f.read())
        for i in models:
            print(f"Using {i} ({models[i]})")
            if models[i].loaded:
                if not titled:
                    models[i].standard(file)
                    titled=True
                output=models[i](file)
                print(i+':', output)
