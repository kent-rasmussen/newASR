#!/usr/bin/env python3
# coding=UTF-8
# This was originally based on https://huggingface.co/blog/fine-tune-w2v2-bert,
# though it has been generalized and adapted from other sources, as well.

from transformers import TrainingArguments, Trainer, AutoModelForCTC
from transformers import Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
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
class Processor(Wav2Vec2BertProcessor,train.Processor):
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
    def from_pretrained(self,*args,**kwargs):
        # print(args,kwargs)
        return self.processor_parent_fn.from_pretrained(*args,**kwargs)
    def __init__(self,**kwargs):
        kwargs['processor_parent_fn']=Wav2Vec2BertProcessor
        train.Processor.__init__(self,**kwargs)
        # Wav2Vec2BertProcessor.__init__()
class TrainWrapper(object):
    def get_data(self):
        self.data=Data(**self.names.datakwargs())
    def get_processor(self):
        processor=self.names.processor_fn(**self.names.processorkwargs())
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
                self.data=Data(**self.names.datakwargs())
            processor.do_prepare_dataset(self.data)
            self.data.cleanup() #data temp files
        if getattr(self.names,'push_to_hub',False):
            print(f"Going to push to {self.names.fqmodelname} repo")
            processor.push_to_hub(self.names.fqmodelname)
        #Processor object goes away at this point
        self.processor=processor.processor
    def get_base_model(self):
        model=train.BaseModel(
                        vocab_size=len(self.processor.tokenizer),
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        **self.names.modelkwargs()
                        )
        #BaseModel object goes away at this point
        self.model=model.model
    def compute_metrics(self,pred):
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
    def train(self):
        self.trainer.train()
        self.model.save_pretrained(self.names.fqmodelname_loc)
    def push(self):
        self.trainer.push()
    def infer(self):
        import infer
        fqmodelnames_loc=[names.fqmodelname_loc]
        models=infer.InferDict(fqmodelnames_loc,checkpoints='infer_checkpoints')
        if not my_options.args.get('audio'):
            if names.language['iso'] == 'gnd': #set a few defaults for test languages
                my_options.args['audio']=[
                    '/home/kentr/Assignment/Tools/WeSay/gnd/ASR/'
                    'Listen_to_Bible_Audio_-_Mata_2_-_Bible__Zulgo___gnd___'
                    'Audio_Bibles-MAT_2_min1.wav'
                    ]
        for file in my_options.args.get('audio'):
            show_standard=True #just once per audio
            for m in models:
                print(os.path.basename(m)+':', models[m](file,show_standard))
                show_standard=False #just once each time
    def __init__(self,model_type,trainer_type,my_options_args):
        self.names=train.Nomenclature(
            **model_type,
            **trainer_type,
            **my_options #pull in default and user settings
            )
        if not isinstance(self.names,train.Nomenclature):
            print(f"Found ({type(names)}) names object; errors may follow.")
            self.names=train.Nomenclature()
        self.data=Data(**self.names.datakwargs())
        if getattr(self.names,'debug',False):
            for attr in dir(self.names):
                if '__' not in attr:
                    print(attr,getattr(self.names,attr))
        if (getattr(self.names,'train',False) or
            getattr(self.names,'push_to_hub',False)):
            self.get_data()
            self.get_processor()
            self.get_base_model()
            data_collator = self.names.data_collator_fn(
                                                    processor=self.processor,
                                                    padding=True)
            #in compute_metrics only:
            self.metric = evaluate.load(self.names.metric_name)
            self.trainer=train.Training(
                        model=self.model,
                        processor=self.processor, #downloads or builds above
                        data=self.data.dbd,
                        data_collator=data_collator,
                        compute_metrics=self.compute_metrics,
                        **self.names.trainingkwargs()
                        )
            if getattr(self.names,'train',False):
                self.trainer.train()
            if getattr(self.names,'push_to_hub',False): #token import in train.py
                self.trainer.push()
        if getattr(self.names,'demo',False):
            if not hasattr(self,'processor'):
                self.get_processor()
            if self.names.fqmodelname_loc:
                train.Demo(self.names)
        if getattr(self.names,'infer',False):
            self.infer()
        # self.trainer=train.Training(**training_kwargs)
        # self.trainer.train()
        # self.trainer.push()
@dataclass
class DataCollatorCTCWithPadding:
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
def notify_user_todo():
    todo=[m for m in ['train','demo','infer'] if my_options.args[m]]
    if len(todo) > 2:
        todo=[', '.join(todo[:-1]),todo[-1]] #just the last
    if len(todo) > 1:
        todo.insert(-1,'and')
    todo=' '.join(todo)
    print(f"going to {todo if todo else 'nothing?!?'}")
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
            'cache_dir':'/media/kentr/Backups/hfcache',
            'dataset_code':'csv',
            'data_file_prefixes':['lexicon_13','examples_300'],
            'data_file_location':'training_data',
            'infer_checkpoints':True,
            # 'train':True,
            'infer':True,
            'refresh_data':True,
            # 'remake_processor':True, #
            # 'reload_model':True
        })
    if 'google.colab' in sys.modules:
        print("It looks like we're running in a colab instance, so setting "
            "some variables now.")
        my_options.args.update({
                'cache_dir':'.',
            })
    my_options.sanitize()
    notify_user_todo()
    model_type={
                'fqbasemodelname':"facebook/w2v-bert-2.0",
                'getmodel_fn':Wav2Vec2BertForCTC, #for tuned models
                'tokenizer_fn':Wav2Vec2CTCTokenizer,
                'feature_extractor_fn':SeamlessM4TFeatureExtractor,
                # 'processor_fn':Wav2Vec2BertProcessor,
                'processor_fn':Processor,
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
                # 'eval_steps':5,
                'logging_steps':5,
                'save_total_limit':2,
                'num_train_epochs':3,
                # 'compute_metrics' is hardcoded; add flexibility if needed
                }
    TrainWrapper(model_type,trainer_type,my_options.args)
    exit()
    for my_options.args['data_file_prefixes'] in [
                                        ['lexicon_640','examples_300'],
                                    ]:
        for my_options.args['metric_name'] in ["wer"]:
            TrainWrapper(model_type,trainer_type,my_options.args)
