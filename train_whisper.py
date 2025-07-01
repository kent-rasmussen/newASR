#!/usr/bin/env python3
# coding=UTF-8
#!/usr/bin/env python3.12
#3.11 until we get bz2 working in 3.12

#pip install --upgrade pip
#pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio

#https://huggingface.co/blog/fine-tune-whisper
# from huggingface_hub import notebook_login
#
# notebook_login()
# import bz2
# exit()
# from datasets import load_dataset, DatasetDict, concatenate_datasets, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
#for DataCollatorSpeechSeq2SeqWithPadding:
from dataclasses import dataclass
import evaluate
import torch
from typing import Any, Dict, List, Union
import sys
import train #this is mine

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class Processor(WhisperProcessor,train.Processor):
    def from_pretrained(self,*args,**kwargs):
        # print(args,kwargs)
        return self.processor_parent_fn.from_pretrained(*args,**kwargs)
    def __init__(self,**kwargs):
        kwargs['processor_parent_fn']=WhisperProcessor
        # These should come from names
        self.tokenizer_fn=kwargs['tokenizer_fn']
        self.fqbasemodelname=kwargs['fqbasemodelname']
        sister_language=kwargs.pop('sister_language')
        self.tokenizer_fn_kwargs={'language':sister_language['name'],
                                    'task':"transcribe"}
        self.make_tokenizer()
        train.Processor.__init__(self,**kwargs)
        # 'tokenizer_fn':WhisperTokenizer,
        # 'feature_extractor_fn':WhisperFeatureExtractor,

class TrainWrapper(train.TrainWrapper):
    # def get_base_model(self):
    #     model=train.BaseModel(
    #                     vocab_size=len(self.processor.tokenizer),
    #                     pad_token_id=self.processor.tokenizer.pad_token_id,
    #                     **self.names.modelkwargs()
    #                     )
    #     #BaseModel object goes away at this point
    #     self.model=model.model
    def compute_metrics(self,pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        error = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {self.names.metric_name: error}
        # return {"wer": wer}
    # def get_processor(self):
    #     processor=self.names.processor_fn(**self.names.processorkwargs())
    #     #any other processing goes here:
    #     processor.do_prepare_dataset(self.data)
    #     self.processor=processor.processor
    def set_whisper_langs(self):
        # print("possible language codes: "
        # f"{[i.strip('<>|') for i in self.model.generation_config.lang_to_id.keys()]}")
        self.model.generation_config.language = self.names.sister_language['mcv_code']
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        # self.model.generation_config.use_cache=True
        self.model.generation_config.use_cache=self.names.use_cache_in_training
    def __init__(self,model_type,trainer_type,my_options_args):
        # past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)
        self.get_names(model_type,trainer_type,my_options_args)
        self.init_debug()
        self.processor_fn_kwargs=self.names.processorkwargs()
        self.get_data_processor_model()
        self.set_whisper_langs()
        self.collator_fn_kwargs={'processor':self.processor,
            'decoder_start_token_id':self.model.config.decoder_start_token_id}
        self.get_trainer()
        super().__init__()
        #in compute_metrics only:
        self.metric = evaluate.load(self.names.metric_name)
        self.do_stuff()
def clear_unused_args(x):
    for arg in ['attention_dropout',
                'hidden_dropout',
                'feat_proj_dropout',
                'mask_time_prob',
                'layerdrop',
                'ctc_loss_reduction',
                ]:
        x.pop(arg)
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
            'dataset_code':'mcv',
            'data_splits':['train'],
            'language_iso':'hau',
            # 'dataset_code':'csv',
            # 'language_iso':'gnd',
            # 'sister_language_iso': 'hau',
            'sister_language_iso': 'ha',
            'cache_dir':'/media/kentr/hfcache',
            # 'data_file_prefixes':['lexicon_640'],#,'examples_4589'
            # 'data_file_prefixes':['lexicon_13'],#,'examples_4589'
            # 'data_file_location':'training_data',
            'infer_checkpoints':True,
            'train':True,
            'infer':True,
            # 'refresh_data':True,
            # 'remake_processor':True, #
            # 'reload_model':True
        })
    clear_unused_args(my_options.args)
    if 'google.colab' in sys.modules:
        print("It looks like we're running in a colab instance, so setting "
            "some variables now.")
        my_options.args.update({
                'cache_dir':'.',
            })
    my_options.sanitize() # wait until everyting is set to do this
    model_type={
                'fqbasemodelname':"openai/whisper-tiny", #small,large, medium, base, large-v3-turbo
                # "efficient-speech/lite-whisper-large-v3-turbo"
                'getmodel_fn':WhisperForConditionalGeneration, #for tuned models
                'tokenizer_fn':WhisperTokenizer,
                'feature_extractor_fn':WhisperFeatureExtractor,
                'processor_fn':Processor,
                }

    trainer_type={
                'data_collator_fn':DataCollatorSpeechSeq2SeqWithPadding,
                'training_args_fn':Seq2SeqTrainingArguments,
                'trainer_fn':Seq2SeqTrainer,
                'learning_rate':1e-5,
                'per_device_train_batch_size':16,
                'save_steps':20,
                # load_best_model_at_end requires the save and eval strategy to match
                # 'eval_strategy': 'epoch',
                # 'save_strategy': 'epoch',
                'eval_strategy': 'steps',
                'save_strategy': 'steps',
                'eval_steps':5,
                'logging_steps':20,
                'save_total_limit':6,
                'num_train_epochs':9,
                # 'compute_metrics' is hardcoded; add flexibility if needed
                }
    for my_options.args['metric_name'] in ['wer','cer']:
        TrainWrapper(model_type,trainer_type,my_options.args)
    exit()
    for my_options.args['data_file_prefixes'] in [
                                        ['lexicon_640','examples_300'],
                                    ]:
        TrainWrapper(model_type,trainer_type,my_options.args)
