#!/usr/bin/env python3
# coding=UTF-8
from typing import Any, Dict, List, Union
from dataclasses import dataclass
#The next two are for the DataCollators:
import torch
from transformers import Wav2Vec2BertProcessor,Wav2Vec2Processor

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        # input_features!
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

@dataclass
class DataCollatorBertCTCWithPadding:
    """from Wav2Vec2BertProcessor source code:
    [`Wav2Vec2Processor` (sic; probably should be `Wav2Vec2BertProcessor`)] offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and [`PreTrainedTokenizer`]."""
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        # input_features!
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

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    from Wav2Vec2Processor source code:
    [`Wav2Vec2Processor`] offers all the functionalities of
    [`Wav2Vec2FeatureExtractor`] and [`PreTrainedTokenizer`].

    from Wav2Vec2BertProcessor source code:
    [`Wav2Vec2Processor` (sic; probably should be `Wav2Vec2BertProcessor`)]
    offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and
    [`PreTrainedTokenizer`].
    """

    processor: (Wav2Vec2Processor,Wav2Vec2BertProcessor)
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        # source had input_values instead of input_features, which throws error?
        # input_values!
        input_features = [{"input_values": feature["input_values"]} for feature in features]
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

def w2v_bert_2():
    from transformers import (Wav2Vec2BertForCTC,
        Wav2Vec2CTCTokenizer,
        SeamlessM4TFeatureExtractor,
        TrainingArguments,
        Trainer
    )
    return {
            'fqbasemodelname':"facebook/w2v-bert-2.0",
            'getmodel_fn':Wav2Vec2BertForCTC, #for tuned models
            'tokenizer_fn':Wav2Vec2CTCTokenizer,
            'feature_extractor_fn':SeamlessM4TFeatureExtractor,
            # 'processor_fn':Wav2Vec2BertProcessor,
            # 'processor_fn':Processor,
            'processor_parent_fn':Wav2Vec2BertProcessor,
            'data_collator_fn':DataCollatorCTCWithPadding,
            'training_args_fn':TrainingArguments,
            'trainer_fn':Trainer,
            'compute_metrics_fn_name':'compute_metrics_ctc',
            'warmup_steps':500, #?
            #These are not needed, as overfitting on minimal data isn't a concern
            'attention-dropout':0.0,
            'hidden-dropout':0.0,
            'feat-proj-dropout':0.0,
            'mask-time-prob':0.0,
            'layerdrop':0.0,
            'ctc-loss-reduction':'mean'
        }
def mms():
    from transformers import (Wav2Vec2ForCTC,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
        TrainingArguments,
        Trainer
    )
    return {
            'fqbasemodelname':"facebook/mms-1b-all",
            'getmodel_fn':Wav2Vec2ForCTC, #for tuned models
            # 'train_adaptor_only':True,
            'tokenizer_fn':Wav2Vec2CTCTokenizer,
            'feature_extractor_fn':Wav2Vec2FeatureExtractor,
            # 'processor_fn':Wav2Vec2BertProcessor,
            # 'processor_fn':Processor,
            'processor_parent_fn':Wav2Vec2Processor,
            'data_collator_fn':DataCollatorCTCWithPadding,
            'training_args_fn':TrainingArguments,
            'trainer_fn':Trainer,
            'ignore_mismatched_sizes':True,
            'compute_metrics_fn_name':'compute_metrics_ctc',
            'make_vocab':True,
            'warmup_steps':100, #500
            #These are not needed, as overfitting on minimal data isn't a concern
            'attention-dropout':0.0,
            'hidden-dropout':0.0,
            'feat-proj-dropout':0.0,
            'mask-time-prob':0.0,
            'layerdrop':0.0,
            'ctc-loss-reduction':'mean'
        }
def whisper():
    from transformers import (WhisperForConditionalGeneration,
        WhisperTokenizer,
        WhisperFeatureExtractor,
        WhisperProcessor,
        # DataCollatorSpeechSeq2SeqWithPadding,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer
    )
    return {
            # 'fqbasemodelname':"openai/whisper-tiny", #small,large, medium, base, large-v3-turbo
            'fqbasemodelname':'openai/whisper-large-v3',
            # "efficient-speech/lite-whisper-large-v3-turbo"
            'getmodel_fn':WhisperForConditionalGeneration, #for tuned models
            'tokenizer_fn':WhisperTokenizer,
            'feature_extractor_fn':WhisperFeatureExtractor,
            # 'processor_fn':Processor,
            'processor_parent_fn':WhisperProcessor,
            'data_collator_fn':DataCollatorSpeechSeq2SeqWithPadding,
            'training_args_fn':Seq2SeqTrainingArguments,
            'trainer_fn':Seq2SeqTrainer,
            'warmup_steps':500, #?
            'predict_with_generate':True,
            'compute_metrics_fn_name':'compute_metrics_whisper',
            'use_cache_in_training': True,
        }
