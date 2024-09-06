
import torch
from logging import getLogger

from VRPEnv import VRPEnv as Env
from VRPModel import VRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from VRProblemDef import generate_task_set

from utils.utils import *


class VRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 meta_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)

            pretrained_dict = checkpoint['model_state_dict']
            
            # pretrained_encoder_dict = {key: value for key, value in pretrained_dict.items() if ('encoder' in key )}

            # for key, value in pretrained_dict.items():
            #     if ('Wq_last' in key ):
            #         print(key)
            #         print(value[1])
            # input()
            model_dict = self.model.state_dict()

            model_dict.update(pretrained_dict)

            #self.model.load_state_dict(checkpoint['model_state_dict'])
            #self.model.load_state_dict(model_dict)

            # for param in self.model.encoder.parameters():
            #     # print(param.requires_grad)
            #     param.requires_grad = False

            self.model.load_state_dict(model_dict)
            for param in self.model.encoder.parameters():
                # print(param.requires_grad)
                param.requires_grad = False
            for param in self.model.decoder.parameters():
                # print(param.requires_grad)
                param.requires_grad = False

        # utility
        self.time_estimator = TimeEstimator()

        self.meta_params = meta_params
        assert self.meta_params['data_type'] == "size_distribution", "Not supported, need to modify the code!"

        self.task_set = generate_task_set(self.meta_params)

    def run(self):
        
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # Save Image
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        reduced_sim_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        task_param_id = int (epoch % len(self.task_set))
        task_param = self.task_set[task_param_id]
        #task_param = [150,0,0]
        #task_param[0] = 200
        if task_param[0] >= 150:
            self.trainer_params['train_batch_size'] = 16
        elif task_param[0] >= 100:
            self.trainer_params['train_batch_size'] = 32
        else:
            self.trainer_params['train_batch_size'] = 64
        #print(task_param)
        self.env.pomo_size = task_param[0]
        self.env.problem_size = task_param[0]

        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss, reduced_sim, prompt_id = self._train_one_batch(batch_size,task_param)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            reduced_sim_AM.update( reduced_sim, 1)

            episode += batch_size



            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},  Sim: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg, reduced_sim_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Sim: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg, reduced_sim_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size, task_param):

        # Prep
        ###############################################
        self.model.train()

        self.env.load_problems(batch_size,task_param)
        reset_state, _, _ = self.env.reset()
        reduced_sim, prompt_id = self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        # reward_mean = reward.float().mean(dim=1, keepdims=True)
        # advantage = torch.div((reward - reward_mean),-reward_mean) # normalize different probelms have different level of rewards
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        #loss_mean = loss_mean - 0.1*reduced_sim

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()


        #print(torch.sum(self.model.promptpool.prompt,dim=2))

        self.optimizer.step()

        #print("after step : ")
        #print(torch.sum(self.model.promptpool.prompt_key,dim=1))
        #print(torch.sum(self.model.promptpool.prompt,dim=2))
        #print(self.model.promptpool.prompt[0][0])
        
        #print(torch.sum(self.model.promptpool.prompt_key,dim=1))
        #print(torch.sum(self.model.promptpool.prompt,dim=2))
        #print(" === ===== ")

        #print(prompt_id)

        return score_mean.item(), loss_mean.item(), reduced_sim, prompt_id
