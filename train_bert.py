#!/usr/bin/env python3
# coding=UTF-8

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
    from model_configs import mms
    # model_type=w2v_bert_2()
    model_type=mms()
    model_type.update({'fqbasemodelname':"facebook/mms-1b-all"})
    print(f"working with model {model_type['fqbasemodelname']}")
    """This is general settings"""
    my_options.args.update({
                            'cache_dir':'/media/kentr/hfcache',
                            'data_file_location':'training_data',
                            'train':True,
                            # 'infer':True,
                            # 'infer_checkpoints':True,
                            'refresh_data':True, #any case w/new processor
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
    print(my_options.args)
    trainer_type={
                'gradient_checkpointing':True,
                # 'learning_rate':1e-5,
                'learning_rate':1e-3,
                # 'per_device_train_batch_size':16,
                # 'per_device_train_batch_size':32,
                'per_device_train_batch_size':64,
                # 'save_steps':20,
                # load_best_model_at_end requires the save and eval strategy to match
                'eval_strategy': 'epoch',
                'save_strategy': 'epoch',
                # 'eval_steps':5,
                # 'logging_steps':20,
                'save_total_limit':3,
                'num_train_epochs':12,
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
    for my_options.args['metric_name'] in ['cer','wer']:
        train.TrainWrapper(model_type,trainer_type,my_options)
        del trainer_type['resume_from_checkpoint']
    exit()
    for my_options.args['data_file_prefixes'] in [
                                        ['lexicon_640','examples_300'],
                                    ]:
        TrainWrapper(model_type,trainer_type,my_options.args)
