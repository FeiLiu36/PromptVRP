##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import pickle

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from VRPTrainer import VRPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_type': 'CVRP',
    'problem_size': 50, # not used
    'pomo_size': 50, # not used
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-2,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [1000,2000,8000],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 10000,
    'train_episodes': 1000,
    'train_batch_size': 64,
    'prev_model_path': None,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_CVRP_50.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './Frozen',  # directory path of pre-trained frozen model and log files saved.
        'epoch': 10000,  # epoch version of pre-trained model to laod.
    }
}

meta_params = {
    # 'enable': True,  # whether use meta-learning or not
    # 'curriculum': True,  # adaptive sample task
    # 'meta_method': 'maml',  # choose from ['maml', 'fomaml', 'maml_fomaml', 'reptile']
    'data_type': 'size_distribution',  # choose from ["size", "distribution", "size_distribution"]
    # 'epochs': 250000,  # the number of meta-model updates: (250*100000) / (1*5*64)
    # 'early_stop_epoch': 50000,  # switch from maml to fomaml
    # 'B': 1,  # the number of tasks in a mini-batch
    # 'k': 1,  # gradient decent steps in the inner-loop optimization of meta-learning method
    # 'L': 0,  # bootstrap steps
    # 'meta_batch_size': 64,  # will be divided by 2 if problem_size >= 150
    # 'update_weight': 100,  # update weight of each task per X iters
    # 'sch_epoch': 225000,  # for the task scheduler of size setting, where sch_epoch = 0.9 * epochs
    # 'solver': 'lkh3_offline',  # solver used to update the task weights, choose from ["bootstrap", "lkh3_online", "lkh3_offline", "best_model"]
    # 'alpha': 0.99,  # params for the outer-loop optimization of reptile
    # 'alpha_decay': 0.999,  # params for the outer-loop optimization of reptile
    # 'beta': 0.9,  # loss weight
}

logger_params = {
    'log_file': {
        'desc': 'train_promptVRP',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()


    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,
                      meta_params=meta_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
