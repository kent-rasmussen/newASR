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

class Trainer(object):
    def test_output(self):
        input_str = self.dbd["train"][0]["sentence"]
        labels = self.tokenizer(input_str).input_ids
        decoded_with_special = self.tokenizer.decode(labels,
                                                    skip_special_tokens=False)
        decoded_str = self.tokenizer.decode(labels, skip_special_tokens=True)

        print(f"Input:                 {input_str}")
        print(f"Decoded w/ special:    {decoded_with_special}")
        print(f"Decoded w/out special: {decoded_str}")
        print(f"Are equal:             {input_str == decoded_str}")
        print(self.dbd["train"][0])
    def __init__(self):
        training_kwargs={
                        'dataset_code': "mcv",
                        'lang_code': "ig",
                        'dataset_dir': "/media/kentr/hfcache/datasets",
                        'cache_dir': "/media/kentr/hfcache/hub",
                        'fqmodelname': "openai/whisper-tiny",
                        # 'fqmodelname': "openai/whisper-small",
                        'getmodel_fn': WhisperForConditionalGeneration,
                        'metricname': "cer",
                        'feature_extractor_fn': WhisperFeatureExtractor,
                        # 'tokenizer_fn': WhisperTokenizer,
                        'processor_fn': WhisperProcessor,
                        'data_collator_fn': DataCollatorSpeechSeq2SeqWithPadding,
                        'training_args_fn': Seq2SeqTrainingArguments,
                        'trainer_fn': Seq2SeqTrainer,
                        # max_steps=5000,
                        # save_strategy="steps",
                        # per_device_eval_batch_size=8,
                        # predict_with_generate=True,
                        # generation_max_length=225,
                        # save_steps=1000,
                        # eval_steps=1000,
                        #     logging_steps=25,
                        # 'peft': True
                    }
        trainer=train.Training(**training_kwargs)
        trainer.train()
        # trainer.push()
        # trainer.demo()
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

class Infer():
    def __call__(self,x):
        audio_dataset = Dataset.from_dict({"audio": [x]}).cast_column("audio",
                                            Audio(sampling_rate=16000))
        sampling_rate = audio_dataset.features["audio"].sampling_rate
        inputs = self.processor(audio_dataset[0]["audio"]["array"],
                                sampling_rate=sampling_rate,
                                return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)
        # print(audio_dataset.features["audio"])
        # print(audio_dataset[0]["audio"])
        # print(sampling_rate)
        # return sampling_rate
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #     return self.processor.decode(outputs)
        # speech_array, sampling_rate = torchaudio.load(file, normalize = True)
        # if sampling_rate != 16000:
        #     transform = torchaudio.transforms.Resample(sampling_rate, 16000)
        #     speech_array = transform(speech_array)
        # speech_array, sampling_rate = read_audio_data(all_test_samples[index][0])
        # self.pipe(speech_array)
        # inputs = self.processor(x, sampling_rate=16_000, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = self.model(**inputs).logits
        # ids = torch.argmax(outputs, dim=-1)[0]
        # return self.processor.decode(ids)
    def __init__(self,repo_name,repo_hf,file):
        import faster_whisper
        model = faster_whisper.WhisperModel(repo_hf)
        segs, _ = model.transcribe(file)
        text = ' '.join(s.text for s in segs)
        print(f'Transcribed text: {text}')
        exit()
        processor_fn=WhisperProcessor
        model_fn=WhisperForConditionalGeneration
        decoder_input_ids or decoder_inputs_embeds
        self.processor = processor_fn.from_pretrained(repo_hf)
        try:
            self.model = model_fn.from_pretrained(
                                            # "automatic-speech-recognition",
                                            repo_name,
                                            # tokenizer=self.processor
                                                )
        except OSError: #i.e., not on local filesystem
            self.model = model_fn.from_pretrained(repo_hf)
        # self.pipe = pipeline("automatic-speech-recognition",
        #                     processor=self.processor,
        #                     tokenizer=self.processor.tokenizer,
        #                     model=repo_name,
        #                     model_kwargs={"target_lang": "fra",
        #                                 "ignore_mismatched_sizes": True})
class Processor(WhisperProcessor,train.Processor):
    def from_pretrained(self,*args,**kwargs):
        # print(args,kwargs)
        return self.processor_parent_fn.from_pretrained(*args,**kwargs)
    def __init__(self,**kwargs):
        kwargs['processor_parent_fn']=WhisperProcessor
        kwargs.pop('tokenizer_fn')
        kwargs.pop('feature_extractor_fn')
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
        self.model.generation_config.language = self.names.sister_language_iso
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
    def __init__(self,model_type,trainer_type,my_options_args):
        # past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)
        self.get_names(model_type,trainer_type,my_options_args)
        self.init_debug()
        self.processor_fn_kwargs=self.names.processorkwargs()
        self.get_data_processor_model()
        self.set_whisper_langs()
        self.collator_fn_kwargs={
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
            'language_iso':'gnd',
            'sister_language_iso': 'hau',
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
                'eval_strategy': 'epoch',
                'save_strategy': 'epoch',
                # 'eval_steps':5,
                'logging_steps':20,
                'save_total_limit':6,
                'num_train_epochs':9,
                # 'compute_metrics' is hardcoded; add flexibility if needed
                }
    for my_options.args['metric_name'] in ['cer','wer']:
        TrainWrapper(model_type,trainer_type,my_options.args)
    exit()
    for my_options.args['data_file_prefixes'] in [
                                        ['lexicon_640','examples_300'],
                                    ]:
        TrainWrapper(model_type,trainer_type,my_options.args)
