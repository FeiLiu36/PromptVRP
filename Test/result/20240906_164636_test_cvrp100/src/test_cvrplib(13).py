##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from VRPTester import VRPTester as Tester


##########################################################################################
# parameters

env_params = {
    #'problem_type': 'CVRP',
    'problem_size': 50,
    'pomo_size':50,
}

model_params = {
    'pool_size': 16, # size of prompt pool
    'top_k': 1, # try the top k prompts
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../Pretrained/',  # directory path of pre-trained model and log files saved.
        'epoch': 10000,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 1*1000,
    'test_batch_size': 500,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size':25,
    

    'test_set_path': '../../Instances/TrainingDistributions/cvrp200_gaussian_mixture_50_7.pkl',

} 
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


fine_tune_params = {
    'enable': False,  # evaluate few-shot generalization
    'fine_tune_episodes': 1000,  # how many data used to fine-tune the pretrained model
    'k': 10,  # fine-tune steps/epochs
    'fine_tune_batch_size': 10,  # the batch size of the inner-loop optimization
    'augmentation_enable': True,
    'optimizer': {
        'lr': 1e-4 * 0.1,
        'weight_decay': 1e-6
    }
}

logger_params = {
    'log_file': {
        'desc': 'test_cvrp100',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    dir_input = '../Instances/CVRPLib/A/'

    dir_output = 'A_result.txt'
    file_out = open(dir_output,'w')
    file_out.write("ins_name    dis    dis_aug"+"\n")
    file_out.close()

    for root,dirs,files in os.walk(dir_input):
        filenameList = files

    #filenameList = "Vrp-Set-Solomon/C101.txt"
    for filename in filenameList:

        if (filename.split('.')[1]=='vrp'):
        
            create_logger(**logger_params)
            _print_config()

            tester_params['test_set_path'] = dir_input+filename
            tester_params['dir_output'] = dir_output

            file = open(tester_params['test_set_path'],'r')
            file.readline()
            file.readline()
            file.readline()
            
            line = file.readline()
            #print(name)
            linesplit = line.split()

            env_params['problem_size'] = int(linesplit[2])-1
            env_params['pomo_size'] = int(linesplit[2])-1

            tester = Tester(env_params=env_params,
                            model_params=model_params,
                            tester_params=tester_params,
                            fine_tune_params=fine_tune_params)

            copy_all_src(tester.result_folder)

            tester.run()

    # tester = Tester(env_params=env_params,
    #                   model_params=model_params,
    #                   tester_params=tester_params,
    #                   fine_tune_params=fine_tune_params)

    # copy_all_src(tester.result_folder)

    # tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
