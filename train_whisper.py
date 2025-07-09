#!/usr/bin/env python3
# coding=UTF-8
if __name__ == '__main__':
    import sys
    import train #this is mine
    if not train.in_venv():
        print("this script is mean to run in a virtual environment, but isn't.")
    from model_configs import whisper
    model_type=whisper()
    model_type.update({'fqbasemodelname':
                    "kent-rasmussen/whisper-large-v3-cer-hau-Hausa_mcv11x9"})
    """ Also: openai/whisper-tiny, openai/whisper-small, openai/whisper-large,
    openai/whisper-large-v2, openai/whisper-medium, openai/whisper-large-v3 """

    my_options=train.options.Parser('train','infer')
    my_options.args.update({ """This is general settings"""
                            'cache_dir':'/media/kentr/hfcache',
                            'data_file_location':'training_data',
                            'train':True,
                            # 'train_adaptor_only':True, #dunno how whisper
                            # 'infer':True,
                            # 'infer_checkpoints':True,
                            # 'refresh_data':True, #any case w/new processor
                            # 'remake_processor':True, #any case if not found
                            # 'reload_model':True #large download!
                            })
    """Pick one of the following three:"""
    # my_options.args.update(train.options.minimal_zulgo_test_options())
    # my_options.args.update({
    #                         'dataset_code':'mcv17',
    #                         'data_splits':['train','validation'],
    #                         'language_iso':'hau',
    #                         'sister_language_iso': 'hau',
    #                         # 'max_data_rows':20,
    #                         })
    my_options.args.update({
                            'dataset_code':'csv',
                            'language_iso':'gnd',
                            'data_file_prefixes':['lexicon_640'],
                            # 'data_file_prefixes':['examples_4589'],
                            # 'data_file_prefixes':['lexicon_13'],
                            })
    trainer_type={
                'learning_rate':1e-3,
                # 'lr_scheduler_type':"linear",
                'per_device_train_batch_size':16,
                # 'save_steps':20,
                # load_best_model_at_end requires the save and eval strategy to match
                'eval_strategy': 'epoch',
                'save_strategy': 'epoch',
                # 'eval_strategy': 'steps',
                # 'save_strategy': 'steps',
                'eval_steps':20,
                'logging_steps':20,
                'save_total_limit':3,
                'num_train_epochs':6,
                'resume_from_checkpoint':True,
                # above if last checkpoint is saved correctly, below if not
                # 'resume_from_checkpoint':os.path.join(
                #                 my_options.args['cache_dir'],
                #                 "mms-1b-all-cer-gnd-Zulgo_640x6/checkpoint-18")
                }
    for my_options.args['metric_name'] in ['wer','cer']:
        train.TrainWrapper(model_type,trainer_type,my_options)
        del trainer_type['resume_from_checkpoint']
    exit()
    for my_options.args['data_file_prefixes'] in [
                                        ['lexicon_640','examples_300'],
                                    ]:
        TrainWrapper(model_type,trainer_type,my_options.args)
