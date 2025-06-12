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
import sys
import datetime
import train #local

def marktime(f,*args,**kwargs):
    def timed(*args,**kwargs):
        start_time=datetime.datetime.now(datetime.UTC)
        r=f(*args,**kwargs)
        print("Function",f.__class__.__name__,datetime.datetime.now(datetime.UTC)-start_time)
        return r
    return timed
class TrainWrapper(object):
    def get_processor(self):
        try:
            print(f"Looking for {self.names.modelname} processor")
            assert getattr(self.names,'remake_processor',False) is False
            self.processor=self.names.processor_fn.from_pretrained(
                                                self.names.fqmodelname_hf,
                                                cache_dir=self.names.cache_dir
                                                            )
            if not self.data.dataset_prepared:
                self.data.show_dimensions_preprocessed()
                self.processor.do_prepare_dataset(self.data)
            print(f"Loaded {self.names.modelname} processor "
                    f"({self.names.fqmodelname})")
        except (Exception,AssertionError) as e:
            if isinstance(e,AssertionError):
                e="by request"
            print(f"Remaking {self.names.modelname} processor ({e})")
            self.processor=train.BertProcessor(**self.names.processorkwargs())
            self.data.show_dimensions_preprocessed()
            self.processor.do_prepare_dataset(self.data)
            self.data.cleanup()
            if getattr(self.names,'push_to_hub',False):
                print(f"Going to push to {self.names.fqmodelname} repo")
                self.processor.push_to_hub(self.names.fqmodelname)
            #make this match try results, a transformers object:
            self.processor=self.processor.processor
    def get_model(self):
        if getattr(self.names,'quant',False):
            import intel_extension_for_pytorch as ipex
            quantization_config=BitsAndBytesConfig(
                # ( load_in_8bit = False
                # llm_int8_threshold = 6.0
                # llm_int8_skip_modules = None
                # llm_int8_enable_fp32_cpu_offload = False )
            )
            # quantization_config=BitsAndBytesConfig(
            #         # load_in_4bit=True,
            #         load_in_8bit = True,
            #         bnb_4bit_quant_type="nf4",         # NormalFloat4 quantization
            #         # bnb_4bit_use_double_quant=True,    # Enable double quantization
            #         # bnb_4bit_compute_dtype=torch.bfloat16 # Compute dtype for faster training
            #         )
            self.quantization_config=quantization_config
        try:
            print(f"Looking for saved {self.names.modelname} model")
            assert getattr(self.names,'remake_model',False) is False
            model=self.names.getmodel_fn.from_pretrained(
                                                self.names.fqmodelname_loc,
                                                cache_dir=self.names.cache_dir
                                                    )
            print(f"Loaded {self.names.modelname} model "
                    f"({self.names.modelname})")
        except (Exception,AssertionError) as e:
            if isinstance(e,AssertionError):
                e="by request"
            print(f"Remaking {self.names.modelname} model ({e})")
            model=train.BaseModel(
                            vocab_size=len(self.processor.tokenizer),
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            **self.names.modelkwargs()
                            )
            #make this match try results: a transformers object
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
    def __init__(self,names): #all options come through names object
        self.names=names
        if not isinstance(self.names,train.Nomenclature):
            print(f"Found ({type(self.names)}) names object; errors will follow.")
            names=train.Nomenclature()
        self.data=train.Data(**{**self.names.datakwargs(), 'make_vocab':True})
        self.get_processor()
        self.get_model()
        data_collator = self.names.data_collator_fn(processor=self.processor,
                                                    padding=True)
        self.metric = evaluate.load(self.names.metric_name) #in compute_metrics only
        self.trainer=train.Training(
                        model=self.model,
                        processor=self.processor, #downloads or builds above
                        data=self.data.dbd,
                        data_collator=data_collator,
                        compute_metrics=self.compute_metrics,
                        **names.trainingkwargs()
                        )
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
def do_run(model_type,trainer_type,options):
    names=train.Nomenclature(
        fqbasemodelname="facebook/w2v-bert-2.0",
        **model_type,
        **trainer_type,
        **options.args #pull in default and user settings
        )
    if options.args.get('debug'):
        for attr in dir(names):
            if '__' not in attr:
                print(attr,getattr(names,attr))
    if options.args.get('train') or options.args.get('push_to_hub'):
        t=TrainWrapper(names)
        if options.args.get('train'):
            t.train()
        if options.args.get('push_to_hub'): #token import in train.py
            t.push()
    if options.args.get('demo'):
        if names.fqmodelname_loc:
            train.Demo(names)
    if options.args.get('infer'):
        import infer
        # from datasets import Dataset, Audio #needed for Infer() only
        print("model:",names.fqmodelname_loc)
        inferer=infer.Infer(names.fqmodelname_loc)
        if not options.args.get('audio'):
            if names.language['iso'] == 'gnd': #set a few defaults for test languages
                audio=['/home/kentr/Assignment/Tools/WeSay/gnd/ASR/'
                    'Listen_to_Bible_Audio_-_Mata_2_-_Bible__Zulgo___gnd___'
                    'Audio_Bibles-MAT_2_min1.wav'
                    ]
        for file in audio:
            if inferer.loaded:
                print(f"Inferring {file}")
                output=inferer(file,show_standard=True)
                print(names.modelname+':', output)
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
    my_options.args.update({
            'language_iso':'gnd',
            'cache_dir':'/media/kentr/Backups/hfcache',
            'dataset_code':'csv',
            'data_file_prefixes':['lexicon_13','examples_300'],
            'data_file_location':'training_data',
            'train':True,
            'refresh_data':True
        })
    if 'google.colab' in sys.modules:
        print("It looks like we're running in a colab instance, so setting "
            "some variables now.")
        my_options.args.update({
                'cache_dir':'.',
            })
    notify_user_todo()
    model_type={
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
                # 'eval_steps':5,
                'logging_steps':5,
                'save_total_limit':2,
                'num_train_epochs':3,
                # 'compute_metrics' is hardcoded; add flexibility if needed
                }
    do_run(model_type,trainer_type,my_options)
    exit()
    for options.args['data_file_prefixes'] in [
                                        ['lexicon_640','examples_300'],
                                    ]:
        for options.args['metric_name'] in ["wer"]:
            do_run(model_type,trainer_type,options)
