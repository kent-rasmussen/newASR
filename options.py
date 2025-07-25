#!/usr/bin/env python3
# coding=UTF-8
version=0.2
import sys
import argparse
def has_argv():
    """This is needed because colab adds an extra -f root argument, meaning
    that it runs with len(sys.argv) = 3, without any visible user arguments
    So this is true when neither in colab, nor without argv specified."""
    val=not bool('google.colab' in sys.modules
                        or len(sys.argv) == 1)#just exe, no other args
    return val
def minimal_zulgo_test_options():
    return {
        'dataset_code':'csv',
        'language_iso':'gnd',
        'data_file_prefixes':['lexicon_13'],#,'examples_4589'
    }
def maximal_zulgo_test_options():
    return {
        'dataset_code':'csv',
        'language_iso':'gnd',
        'data_file_prefixes':['lexicon_640','examples_4589']
    }
def quantization():
    return {
        'load_in_8bit':True,
        'device_map':"auto"
    }
class Parser(object):
    def do_arg_set(self,arg_set):
        self.default_list=getattr(self,arg_set)
    # def parse():
        if has_argv():#'google.colab' in sys.modules: #no sys.argv here
            # print("parsing args!")
            self.parse_argv()
        else:
            self.defaults_only()
        # print(f"done with {arg_set}")
        try:
            self.arg_sets.remove(arg_set)
        except ValueError:
            pass #'base' isn't in this list, ever.
    def sanitize(self):
        if 'google.colab' in sys.modules:
            print("It looks like we're running in a colab instance, so setting "
                "some variables now.")
            self.args.update({
                    'cache_dir':'.',
                    'data_file_location':'/content/drive/MyDrive/training_data'
                    })
        if not self.args.get('cache_dir_tuned'):
            self.args.update({'cache_dir_tuned':self.args.get('cache_dir')})
    def __init__(self,*args):#_sets=set(),parents=[]):
        arg_sets_ok={'base','train','infer'}
        if set(args)-arg_sets_ok:
        # if arg_set not in arg_sets:
            print(f"arg_sets ‘{args}’ not all in arg_sets {arg_sets}")
        self.parents=[]
        self.base = [
            ('-l', '--language-iso',
                {'help':"ISO 639-3 (Ethnologue) code of language",
                # 'required':has_argv()
            }),
            ('-c', '--cache-dir',
                {'help':"where models and data are stored locally"
                #Yes, this is redundant, but matches Huggingface use
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
        self.train = [
            ('-t', '--train',
                {'help':"Train a new ASR model",
                    'action':'store_true'
                }),
            ('-l', '--language-iso',
                {'help':"ISO 639-3 (Ethnologue) code of language",
                # 'required':has_argv()
            }),
            ('-s', '--sister-language-iso',
                {'help':"ISO 639-3 (Ethnologue) code of a related language"
                    "already covered by the model to be fine-tuned",
            }),
            ('--cache-dir-tuned',
                {'help':"Directory to store tuned models",
            }),
            ('--lr-scheduler-type',
                {'help':"Set learning rate scheduler type:"
                    "“linear” → transformers.get_linear_schedule_with_warmup"
                    "“cosine” → transformers.get_cosine_schedule_with_warmup"
                    "“cosine_with_restarts” -->transformers.get_cosine_with_hard_restarts_schedule_with_warmup"
                    "“polynomial” → transformers.get_polynomial_decay_schedule_with_warmup"
                    "“constant” → transformers.get_constant_schedule"
                    "“constant_with_warmup” → transformers.get_constant_schedule_with_warmup"
                    "“inverse_sqrt” → transformers.get_inverse_sqrt_schedule“linear” → transformers.get_linear_schedule_with_warmup"
                    "“cosine” → transformers.get_cosine_schedule_with_warmup"
                    "“cosine_with_restarts” -->transformers.get_cosine_with_hard_restarts_schedule_with_warmup"
                    "“polynomial” → transformers.get_polynomial_decay_schedule_with_warmup"
                    "“constant” → transformers.get_constant_schedule"
                    "“constant_with_warmup” → transformers.get_constant_schedule_with_warmup"
                    "“inverse_sqrt” → transformers.get_inverse_sqrt_schedule",
                'default':'linear'
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
                            "(for downloads and pushing)",
            }),
            ('--metric-name',
                {'help':"Name of metric to evaluate ASR (e.g., "
                    "word error rate; WER)",
                    'choices':['wer','cer'],
                    'default':'wer'
                }),
            ('--remake-processor',
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
                {'help':"make a vocab file for the CTC tokenizer",
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
            ('-m','--reload-model',
                {'help':"Reload Model (even if found cached)",
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
            # ('--attention-dropout',{}),
            # ('--hidden-dropout',{}),
            # ('--feat-proj-dropout',{}),
            # ('--mask-time-prob',{}),
            # ('--layerdrop',{}),
            # ('--ctc-loss-reduction',{})
        ]
        self.infer = [
            ('-i', '--infer',
                {'help':"Infer on a model (get text from audio)",
                    'action':'store_true'
                }),
            ('-a', '--audio',
                {'help':"Audio file to infer (convert to text)",
                    'action':'append',
                    # 'required':False,
                    'default':[]
                })
                ]
        self.arg_sets=list(args)
        self.do_arg_set('base')
        while self.arg_sets:
            self.parents=[self]
            self.do_arg_set(self.arg_sets[0])
        if hasattr(self,'ap'):
            self.ap.finalize()
            self.args = self.ap.args
        # self.sanitize()
    def defaults_only(self):
        """This is used when we want sane defaults, but don't have sys.argv
        (e.g., colab)"""
        def sanify_arg(x):
            return x.strip('-').translate(str.maketrans('-','_'))
        self.args={k:j.args[k] for j in self.parents for k in j.args}
        self.args.update({sanify_arg(arg):kwargs['default']
            for *args,kwargs in self.default_list
            for arg in args
            if '--' in arg
            if 'default' in kwargs
        })
        self.args.update({sanify_arg(arg):False #default!
            for *args,kwargs in self.default_list
            for arg in args
            if '--' in arg
            if 'action' in kwargs
            if kwargs['action'] == 'store_true'
        })
        # return return_args
    def parse_argv(self):
        self.ap= ArgumentParser(options=self.default_list,
                                parents=[i.ap for i in self.parents])
class ArgumentParser(argparse.ArgumentParser):
    def finalize(self):
        self.args = vars(self.parse_args())
        for i in [
                ('special_characters_ok','no_special_characters'),
                ('capital_letters_ok','no_capital_letters')
                ]: #convert first to second
            self.args[i[1]]=not self.args.pop(i[0])
    def __init__(self, options, **kwargs):
        print(f"Argument Parser dealing with kwargs {kwargs}")
        print(f"Argument Parser dealing with options {options}")
        prog='ASR_Trainer'
        description='This module programmatically trains Automatic '
        'Speech Recognition (ASR) modules, for scalable mass '
        'production with minimal training data.'
        '\nUsing various options, one can train, infer and push '
        'all in the same run, if desired.'
        if kwargs['parents']:
            kwargs['add_help']=False
            kwargs['conflict_handler']='resolve'
        super().__init__(prog=prog, description=description, **kwargs)

        for *args,kwargs in options:
            self.add_argument(*args,**kwargs)
        # This section converts various settings from what makes sense to the
        # user to what the computer uses, especially where default is true,
        # rather than false (unspecified)
        # print(f"Found user args {self.args}")
if __name__ == '__main__':
    options=Parser('train','infer')
    print(type(options))
    print(options.args)
    if 'audio' in options.args:
        print('True')
    else:
        print('False')
    exit()

    # options=Parser(parents=[options],arg_set='demo')
    # print(type(options))
    # print(options.args)
