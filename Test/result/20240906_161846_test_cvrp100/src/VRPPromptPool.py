import torch
import torch.nn as nn
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler,normalize

#import matplotlib.pyplot as plt

class Prompt(nn.Module):
    def __init__(self, length=1, embed_dim=256, embedding_key='mean', prompt_init='uniform', prompt_pool=False, load_key = True,
                 prompt_key=False, pool_size=None, top_k=None, prompt_size=3,key_div_bound=100, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()

        self.length = length # length of token, for vrp it is set to be 1
        self.embed_dim = embed_dim # embedding size
        self.prompt_pool = prompt_pool # input a costomized prompt pool
        self.embedding_key = embedding_key # ways for calculating embedding keys
        self.prompt_init = prompt_init # ways for prompt initilization
        self.prompt_key = prompt_key # w
        self.prompt_timesused = None
        self.prompt_timesnotused = None
        self.pool_size = pool_size
        self.top_k = top_k
        self.load_key = load_key
        self.prompt_size = prompt_size
        self.key_div_bound = key_div_bound

        self.batchwise_prompt = batchwise_prompt

        self.scalor = None

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, self.prompt_size*6*embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                # shape: (pool_size, length, embedding_size)
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
                # shape: (pool_size, length, embedding_size)


        # promt_base = torch.tensor([  -0.502 ,  4.582 ,  -1.975 ,  2.436 ,  2.998 ,  -2.090 ,  -2.501 ,  2.510 ,  1.311 ,  1.051 ,  -2.511 ,  2.987 ,  
        #                            -0.216 ,  2.185 ,  0.922 ,  -4.794 ,  -2.690 ,  3.546 ,  0.190 ,  -4.727 ,  -7.016 ,  1.833 ,  -2.942 ,  -1.142 ,  
        #                            2.647 ,  0.423 ,  2.381 ,  -3.117 ,  -5.604 ,  -2.448 ,  -2.421 ,  -4.767 ,  -0.916 ,  -0.769 ,  -2.556 ,  -0.123 ,  
        #                            -1.064 ,  1.365 ,  -5.913 ,  1.741 ,  -2.253 ,  -1.958 ,  -1.673 ,  3.214 ,  -4.336 ,  -2.065 ,  -1.696 ,  1.766 ,  1.436 ,  
        #                            -2.377 ,  0.054 ,  -0.733 ,  -2.417 ,  6.459 ,  -0.073 ,  -0.352 ,  4.489 ,  -0.483 ,  -1.878 ,  -0.748 ,  2.271 ,  3.294 ,  
        #                            -0.591 ,  -4.725 ,  1.855 ,  -2.486 ,  -3.460 ,  2.253 ,  0.860 ,  -2.412 ,  -0.986 ,  -0.005 ,  -1.902 ,  5.011 ,  1.181 ,  
        #                            0.939 ,  -2.782 ,  -3.617 ,  -4.745 ,  3.976 ,  -4.916 ,  0.866 ,  -0.639 ,  -0.255 ,  1.437 ,  -3.996 ,  2.873 ,  -1.171 ,  
        #                            1.873 ,  -2.162 ,  -0.536 ,  0.891 ,  0.753 ,  -4.843 ,  -4.159 ,  -4.096 ,  2.781 ,  3.333 ,  -0.273 ,  -0.520 ,  2.329 ,  
        #                            3.074 ,  -2.031 ,  -0.010 ,  -4.379 ,  -2.451 ,  1.917 ,  0.136 ,  -0.202 ,  3.300 ,  1.518 ,  3.711 ,  2.152 ,  -0.329 ,  
        #                            1.202 ,  2.919 ,  -1.855 ,  1.398 ,  -4.542 ,  0.433 ,  9.188 ,  2.736 ,  0.554 ,  -0.048 ,  2.881 ,  1.277 ,  -0.535 ,  4.020 ],requires_grad=True)
        if self.load_key:

            # n_layer = 6 # number of layers of the pretrained encoder
            # size_embedding = 128 # embedding size of the pretrained encoder

            # filename_input = '../embedding_alllayers_beforeNorm_32.dat'
            # filename_embedding = open(filename_input,'r')
            # lines = filename_embedding.readlines()

            # length = n_layer*size_embedding # total length of embedding feature
            # n_instance = len(lines) # number of instances / samples

            # data = np.zeros((n_instance,length))
            # i = 0
            # for line in lines:
            #     data[i] = line.split()
            #     i = i + 1

            # scalor = StandardScaler()
            # scalor.fit(data)
            # data_scaled = scalor.transform(data)

            # self.scalor = scalor

            # cluster = KMeans(n_clusters=self.pool_size,random_state=2023).fit(data_scaled)

            # pickle.dump(scalor, open("std_scalor","wb"))
            # pickle.dump(cluster, open("kmeans_cluster","wb"))
            #self.scalor = pickle.load(open("std_scalor","rb"))
            #self.cluster = pickle.load(open("kmeans_cluster_"+str(self.pool_size),"rb"))


            #cluster_centers = self.cluster.cluster_centers_



            # for i in range(self.pool_size):
            #     plt.plot(np.arange(length),cluster_centers[i])
            # plt.show()

            #key_shape = (pool_size, 6*embed_dim)
            #self.prompt_key = nn.Parameter(torch.zeros(key_shape))

            keys = pickle.load(open('keys_new_'+str(self.pool_size),'rb'))
            self.prompt_key = torch.tensor(keys).to(torch.device('cuda', 0)).float()

            
            #input()
        # if using learnable prompt keys

        else:
            if prompt_key:
                key_shape = (pool_size, 5*embed_dim)
                if prompt_key_init == 'zero':
                    self.prompt_key = nn.Parameter(torch.zeros(key_shape)+torch.tensor([  0.124 ,  1.036 ,  1.387 ,  3.684 ,  4.764 ,  -3.025 ,  3.117 ,  0.093 ,  -0.620 ,  0.971 ,  1.587 ,  -4.532 ,  -0.227 ,  -0.471 ,  2.889 ,  0.505 ,  -0.044 ,  1.105 ,  1.052 ,  -0.651 ,  -0.538 ,  3.675 ,  2.419 ,  -1.551 ,  0.347 ,  -0.728 ,  2.461 ,  0.514 ,  -2.535 ,  0.621 ,  -0.083 ,  -1.701 ,  -0.928 ,  1.109 ,  -2.026 ,  2.628 ,  0.876 ,  -0.180 ,  -0.180 ,  -1.285 ,  0.579 ,  -0.339 ,  -0.005 ,  -0.470 ,  2.301 ,  -1.302 ,  -0.526 ,  -0.014 ,  1.643 ,  5.364 ,  2.212 ,  -0.484 ,  -3.853 ,  0.818 ,  1.090 ,  0.081 ,  0.642 ,  3.392 ,  -0.119 ,  -1.587 ,  2.153 ,  1.787 ,  -0.006 ,  -1.357 ,  -2.900 ,  -2.816 ,  0.251 ,  -1.397 ,  -0.092 ,  0.795 ,  -0.249 ,  -0.216 ,  0.344 ,  1.206 ,  -0.345 ,  -0.596 ,  -0.051 ,  -2.968 ,  -0.202 ,  0.366 ,  2.685 ,  0.993 ,  -0.380 ,  2.676 ,  -1.187 ,  -0.061 ,  -1.178 ,  1.630 ,  -0.284 ,  -0.523 ,  -1.119 ,  -0.450 ,  -0.369 ,  -0.418 ,  0.447 ,  0.117 ,  0.173 ,  0.135 ,  0.262 ,  -0.042 ,  -0.061 ,  1.150 ,  1.656 ,  -1.606 ,  2.254 ,  
                                                                                        -1.972 ,  0.063 ,  -0.531 ,  -0.272 ,  0.067 ,  -1.680 ,  0.636 ,  -0.880 ,  -0.466 ,  -0.546 ,  0.287 ,  0.685 ,  0.873 ,  0.345 ,  0.120 ,  1.483 ,  -1.598 ,  -3.699 ,  2.823 ,  -0.840 ,  -1.466 ,  0.929 ,  -0.228 ,  -0.613 ,  -0.344 ,  0.478 ,  -0.673 ,  -0.064 ,  -1.576 ,  0.646 ,  -0.114 ,  1.043 ,  0.158 ,  -0.545 ,  -0.167 ,  0.484 ,  0.447 ,  -1.539 ,  -1.020 ,  0.745 ,  -0.001 ,  -0.186 ,  0.077 ,  0.253 ,  -0.727 ,  -0.852 ,  0.962 ,  -1.424 ,  -0.920 ,  0.089 ,  0.832 ,  -0.216 ,  1.075 ,  -0.021 ,  0.355 ,  0.122 ,  0.388 ,  0.061 ,  0.257 ,  -0.141 ,  0.456 ,  0.090 ,  0.388 ,  -1.383 ,  0.281 ,  -0.505 ,  0.576 ,  -0.443 ,  -1.386 ,  -0.450 ,  0.218 ,  -0.574 ,  0.641 ,  -0.103 ,  -0.300 , 
                                                                                        -0.180 ,  -0.052 ,  0.483 ,  -0.435 ,  0.071 ,  -1.483 ,  -0.644 ,  0.103 ,  -0.726 ,  0.190 ,  0.595 ,  0.683 ,  -0.541 ,  0.338 ,  -0.543 ,  -0.653 ,  -1.010 ,  -0.277 ,  0.348 ,  0.340 ,  0.176 ,  0.263 ,  0.502 ,  -0.073 ,  0.374 ,  -0.444 ,  0.394 ,  -0.093 ,  -0.121 ,  -0.748 ,  -0.514 ,  1.306 ,  0.285 ,  -0.708 ,  -0.567 ,  -0.455 ,  -0.215 ,  0.112 ,  -0.578 ,  0.229 ,  -0.469 ,  0.469 ,  -0.068 ,  -0.745 ,  -0.618 ,  -0.610 ,  -0.430 ,  -2.769 ,  -0.070 ,  -0.166 ,  0.117 ,  -0.343 ,  -0.781 ,  1.382 ,  -0.925 ,  0.271 ,  0.148 ,  -0.492 ,  -1.443 ,  0.702 ,  -0.610 ,  0.258 ,  -0.426 ,  1.584 ,  -0.885 ,  0.060 ,  0.114 ,  0.281 ,  -1.270 ,  1.767 ,  1.515 ,  0.113 ,  0.558 ,  0.947 ,  -0.155 , 
                                                                                        -0.106 ,  0.013 ,  -0.075 ,  0.145 ,  -0.491 ,  0.192 ,  0.159 ,  -0.181 ,  0.725 ,  0.201 ,  0.391 ,  -0.471 ,  0.069 ,  -0.144 ,  -0.139 ,  -0.061 ,  0.422 ,  -0.034 ,  -0.029 ,  -0.101 ,  0.128 ,  -0.483 ,  0.180 ,  -0.321 ,  0.253 ,  -0.216 ,  -0.157 ,  -0.191 ,  0.069 , 
                                                                                        -0.296 ,  -0.329 ,  0.048 ,  0.307 ,  0.176 ,  0.035 ,  -0.317 ,  0.169 ,  -0.062 ,  0.694 ,  -0.167 ,  -0.059 ,  -0.166 ,  -0.124 ,  -0.047 ,  -0.008 ,  0.160 ,  0.355 ,  0.071 ,  0.001 ,  0.137 ,  0.088 ,  0.083 ,  0.139 ,  0.361 ,  -0.234 ,  -0.099 ,  0.156 ,  0.270 ,  0.341 , 
                                                                                        -0.528 ,  0.271 ,  0.079 ,  -0.053 ,  0.802 ,  -0.102 ,  -0.055 ,  0.048 ,  0.791 ,  -0.042 ,  0.105 ,  0.103 ,  -0.326 ,  -0.127 ,  0.128 ,  -0.083 ,  -0.399 ,  0.609 ,  -0.010 ,  -0.313 ,  0.196 ,  -0.143 ,  0.178 ,  -0.242 ,  -0.467 ,  -0.097 ,  -0.078 ,  -0.270 ,  0.036 ,  -0.086 ,  -0.074 ,  0.462 ,  0.290 ,  0.278 ,  0.019 ,  -0.146 ,  0.202 ,  0.073 ,  0.492 ,  -0.064 ,  0.102 ,  -0.470 ,  0.070 ,  -0.261 ,  -0.024 ,  0.449 ,  -0.014 ,  -0.115 ,  -0.018 ,  -0.306 ,  0.093 ,  -0.071 ,  0.011 ,  -0.627 ,  0.151 ,  0.578 ,  0.034 ,  0.170 ,  0.026 ,  0.040 ,  -0.164 ,  -0.147 ,  -0.362 ,  -0.083 ,  0.173 ,  -0.775 ,  -0.118 ,  0.716 ,  0.405 ,  -0.105 ,  -0.017 ,  0.229 ,  -0.010 ,  -0.080 ,  -0.111 ,  -0.034 ,  0.211 ,  0.192 ,  0.159 ,  -0.190 ,  0.223 ,  0.146 ,  0.113 ,  0.080 ,  -0.166 ,  -0.275 ,  0.108 ,  -0.304 ,  0.173 ,  0.281 ,  -0.208 ,  -0.292 ,  0.107 ,  -0.032 ,  -0.015 ,  -0.238 ,  -0.144 ,  0.258 ,  0.203 ,  -0.003 ,  0.127 ,  -0.130 ,  -0.016 ,  0.644 ,  0.151 ,  0.070 ,  -0.372 ,  0.295 ,  0.335 ,  0.014 ,  -0.192 ,  0.218 ,  0.222 ,  -0.050 ,  -0.089 ,  0.081 ,  -0.035 ,  0.169 ,  0.070 ,  0.275 ,  -0.126 ,  -0.006 ,  0.236 ,  0.186 ,  -0.387 ,  -0.110 ,  -0.216 ,  -0.125 ,  -0.233 ,  0.273 ,  -0.074 ,  0.108 ,  -0.111 ,  -0.144 ,  
                                                                                        0.080 ,  -0.368 ,  -0.255 ,  0.101 ,  0.135 ,  -0.579 ,  0.417 ,  -0.132 ,  0.017 ,  0.040 ,  0.078 ,  0.242 ,  0.314 ,  -0.169 ,  0.005 ,  -0.087 ,  
                                                                                        0.341 ,  -0.466 ,  -0.298 ,  -0.222 ,  -0.118 ,  -0.347 ,  0.277 ,  -0.329 ,  -0.227 ,  -0.035 ,  -0.119 ,  0.359 ,  0.000 ,  0.169 ,  0.014 ,  -0.421 ,  0.102 ,  0.385 ,  0.191 ,  -0.104 ,  0.185 ,  -0.217 ,  0.050 ,  0.132 ,  0.178 ,  0.253 ,  0.205 ,  0.265 ,  0.010 ,  0.267 ,  0.137 ,  0.019 ,  -0.145 ,  0.033 ,  -0.174 ,  -0.221 ,  0.089 ,  0.047 ,  0.396 ,  0.098 ,  0.026 ,  -0.461 ,  0.036 ,  0.061 ,  0.413 ,  -0.451 ,  -0.111 ,  -0.217 ,  0.123 ,  0.225 ,  0.113 ,  0.088 ,  -0.012 ,  -0.045 ,  -0.069 ,  -0.484 ,  0.855 ,  0.262 ,  0.023 ,  0.374 ,  0.151 ,  0.172 ,  -0.345 ,  0.140 ,  -0.741 ,  -0.215 ,  -0.330 ,  0.419 ,  0.329 ,  0.166 ,  -0.251 ,  -0.247 ,  0.046 ,  0.242 ,  -0.057 ,  -0.089 ,  -0.194 ,  -0.314 ,  0.202 ,  -0.117 ,  0.068 ,  0.429 ,  -0.211 ,  0.106 ,  -0.011 ,  -0.073 ,  -0.039 ,  -0.090 ,  -0.165 ,  0.169 ,  0.287 ,  0.149 ,  0.041 ,  0.098 ,  0.427 ,  0.279 ,  0.194 ,  -0.320 ,  -0.122 ,  0.231 ,  0.022 ,  0.586 ,  -0.260 ,  0.074 ,  0.149 ,  -0.388 ,  0.081 ,  -0.079 ,  -0.487 ,  -0.545 ,  -0.090 ,  -0.157 ,  -0.224 ,  0.147 ,  0.291 ,  0.384 ,  0.067 ,  -0.312 ,  0.337 ,  -0.241 ,  0.129 ,  0.501 ,  0.123 ,  0.235 ,  -0.255 ,  0.020 ,  0.104 ,  0.223 ,  -0.278 ,  0.021 ,  -0.037 ,  
                                                                                        -0.276 ,  -0.038 ,  -0.240 ,  -0.185 ,  0.305 ,  -0.275 ,  -0.501 ,  -0.264 ,  -0.390 ,  0.232 ,  -0.253 ,  -0.112 ,  -0.107 ,  -0.034 ,  0.096 ,  0.247 ,  0.432 ,  0.126 ,  -0.055 ,  0.073 ,  0.214 ,  0.055 ,  0.304 ,  -0.344 ,  0.059 ,  0.288 ,  0.121 ,  0.351 ,  0.233 ,  0.040 ,  -0.189 ,  -0.029 ,  -0.022 ,  0.230 ,  0.063 ,  0.338 ,  -0.333 ,  0.100 ,  0.347 ,  -0.204 ,  -0.141 ,  0.271 ,  -0.104 ,  -0.343 ,  0.564 ,  -0.113 ,  0.451 ,  -0.009 ,  -0.240 ,  0.029 ,  0.182 ,  -0.216 ,  0.014 ,  -0.323 ,  -0.130 ,  0.128 ,  0.074 ,  0.140 ,  -0.042 ,  0.065 ,  -0.110 ,  0.079 ,  -0.045 ,  0.252 ,  -0.011 ,  -0.468 ,  -0.124 ,  0.172 ,  0.156 ,  0.157 ,  -0.081 ,  0.234 ,  0.265 ,  -0.165 ,  -0.170 ,  -0.121 ,  -0.234 ,  0.011 ,  0.013 ,  0.050 ,  0.150 ,  0.026 ,  -0.054 ,  -0.260 ,  0.343 ,  -0.373 ,  0.206 ,  -0.270 ,  0.164 ,  0.230 ,  0.024 ,  0.068 ,  0.165 ,  -0.075 ,  0.145 ,  0.114 ,  0.016 ,  -0.369 ,  -0.192 ,  0.275 ,  -0.341 ,  -0.062 ,  0.235 ,  0.318 ,  -0.134 ,  0.049 ,  -0.022 ,  0.191 ,  0.008 ,  0.598 ,  0.128 ,  0.146 ,  -0.146 ,  0.320 ,  -0.072 , 
                                                                                        -0.137 ,  0.083 ,  -0.138 ,  -0.135 ,  0.397 ,  -0.037 ,  0.068 ,  -0.076 ,  0.084 ,  -0.066 ,  -0.187 ,  -0.186 ,  0.062 ,  0.008 ,  0.268 ,  0.147 ,  0.110 ,  0.046 ,  -0.186 ,  -0.310 ,  -0.428 ,  -0.122 ,  0.197 ,  -0.221 ,  -0.038 ,  -0.251 ,  -0.196 ,  0.254 ,  0.297 ,  0.250 ,  0.104 ,  -0.036 ,  -0.272 ,  -0.226 ,  0.178 ,  -0.225 ,  -0.051 ,  0.062 ,  0.200 ,  0.042 ,  0.053 ,  0.019 ,  -0.100 ,  -0.121 ,  -0.354 ,  0.314 ,  -0.085 ,  0.034 ,  -0.116 ,  0.349 ,  -0.357 ,  -0.031 ,  -0.330 ,  -0.023 ,  0.091 ,  0.104 ,  -0.416 ,  -0.120 , ],requires_grad=True).expand(key_shape))
                elif prompt_key_init == 'uniform':
                #     print(promt_base.expand(key_shape)[0])
                #     print(nn.init.uniform_(nn.Parameter(torch.randn(key_shape)), -1, 1)[0])
                #     input()
                    self.prompt_key = nn.Parameter(nn.init.uniform_(torch.randn(key_shape), -0.5, 0.5) +torch.tensor([  -0.613 ,  -0.344 ,  0.478 ,  -0.673 ,  -0.064 ,  -1.576 ,  0.646 ,  -0.114 ,  1.043 ,  0.158 ,  -0.545 , 
                                                                                                                    -0.167 ,  0.484 ,  0.447 ,  -1.539 ,  -1.020 ,  0.745 ,  -0.001 ,  -0.186 ,  0.077 ,  0.253 ,  -0.727 ,  -0.852 ,  0.962 ,  -1.424 ,  -0.920 ,  0.089 ,  0.832 ,  -0.216 ,  1.075 ,  -0.021 ,  0.355 ,  0.122 ,  0.388 ,  0.061 ,  0.257 ,  -0.141 ,  0.456 ,  0.090 ,  0.388 ,  -1.383 ,  0.281 ,  
                                                                                                                    -0.505 ,  0.576 ,  -0.443 ,  -1.386 ,  -0.450 ,  0.218 ,  -0.574 ,  0.641 ,  -0.103 ,  -0.300 ,  -0.180 ,  -0.052 ,  0.483 ,  -0.435 ,  0.071 ,  -1.483 ,  -0.644 ,  0.103 ,  -0.726 ,  0.190 ,  0.595 ,  0.683 ,  -0.541 ,  0.338 ,  -0.543 ,  -0.653 ,  -1.010 ,  -0.277 ,  0.348 ,  0.340 ,  0.176 ,  
                                                                                                                    0.263 ,  0.502 ,  -0.073 ,  0.374 ,  -0.444 ,  0.394 ,  -0.093 ,  -0.121 ,  -0.748 ,  -0.514 ,  1.306 ,  0.285 ,  -0.708 ,  -0.567 ,  -0.455 ,  -0.215 ,  0.112 ,  -0.578 ,  0.229 ,  -0.469 ,  0.469 ,  -0.068 ,  -0.745 ,  -0.618 ,  -0.610 ,  -0.430 ,  -2.769 ,  -0.070 ,  -0.166 ,  0.117 ,  -0.343 , 
                                                                                                                        -0.781 ,  1.382 ,  -0.925 ,  0.271 ,  0.148 ,  -0.492 ,  -1.443 ,  0.702 ,  -0.610 ,  0.258 ,  -0.426 ,  1.584 ,  -0.885 ,  0.060 ,  0.114 ,  0.281 ,  -1.270 ,  1.767 ,  1.515 ,  0.113 ,  0.558 ,  0.947 ,  -0.155 ,  -0.106 ,  0.013 ,  -0.075 ,  0.145 ,  -0.491 ,  0.192 ,  0.159 ,  -0.181 ,  0.725 , 
                                                                                                                        0.201 ,  0.391 ,  -0.471 ,  0.069 ,  -0.144 ,  -0.139 ,  -0.061 ,  0.422 ,  -0.034 ,  -0.029 ,  -0.101 ,  0.128 ,  -0.483 ,  0.180 ,  -0.321 ,  0.253 ,  -0.216 ,  -0.157 ,  -0.191 ,  0.069 ,  -0.296 ,  -0.329 ,  0.048 ,  0.307 ,  0.176 ,  0.035 ,  -0.317 ,  0.169 ,  -0.062 ,  0.694 ,  -0.167 ,  -0.059 , 
                                                                                                                        -0.166 ,  -0.124 ,  -0.047 ,  -0.008 ,  0.160 ,  0.355 ,  0.071 ,  0.001 ,  0.137 ,  0.088 ,  0.083 ,  0.139 ,  0.361 ,  -0.234 ,  -0.099 ,  0.156 ,  0.270 ,  0.341 ,  -0.528 ,  0.271 ,  0.079 ,  -0.053 ,  0.802 ,  -0.102 ,  -0.055 ,  0.048 ,  0.791 ,  -0.042 ,  0.105 ,  0.103 ,  -0.326 ,  -0.127 ,  0.128 , 
                                                                                                                            -0.083 ,  -0.399 ,  0.609 ,  -0.010 ,  -0.313 ,  0.196 ,  -0.143 ,  0.178 ,  -0.242 ,  -0.467 ,  -0.097 ,  -0.078 ,  -0.270 ,  0.036 ,  -0.086 ,  -0.074 ,  0.462 ,  0.290 ,  0.278 ,  0.019 ,  -0.146 ,  0.202 ,  0.073 ,  0.492 ,  -0.064 ,  0.102 ,  -0.470 ,  0.070 ,  -0.261 ,  -0.024 ,  0.449 ,  -0.014 ,  -0.115 , 
                                                                                                                            -0.018 ,  -0.306 ,  0.093 ,  -0.071 ,  0.011 ,  -0.627 ,  0.151 ,  0.578 ,  0.034 ,  0.170 ,  0.026 ,  0.040 ,  -0.164 ,  -0.147 ,  -0.362 ,  -0.083 ,  0.173 ,  -0.775 ,  -0.118 ,  0.716 ,  0.405 ,  -0.105 ,  -0.017 ,  0.229 ,  -0.010 ,  -0.080 ,  -0.111 ,  -0.034 ,  0.211 ,  0.192 ,  0.159 ,  -0.190 ,  0.223 , 
                                                                                                                                0.146 ,  0.113 ,  0.080 ,  -0.166 ,  -0.275 ,  0.108 ,  -0.304 ,  0.173 ,  0.281 ,  -0.208 ,  -0.292 ,  0.107 ,  -0.032 ,  -0.015 ,  -0.238 ,  -0.144 ,  0.258 ,  0.203 ,  -0.003 ,  0.127 ,  -0.130 ,  -0.016 ,  0.644 ,  0.151 ,  0.070 ,  -0.372 ,  0.295 ,  0.335 ,  0.014 ,  -0.192 ,  0.218 ,  0.222 ,  -0.050 ,
                                                                                                                                    -0.089 ,  0.081 ,  -0.035 ,  0.169 ,  0.070 ,  0.275 ,  -0.126 ,  -0.006 ,  0.236 ,  0.186 ,  -0.387 ,  -0.110 ,  -0.216 ,  -0.125 ,  -0.233 ,  0.273 ,  -0.074 ,  0.108 ,  -0.111 ,  -0.144 ,  0.080 ,  -0.368 ,  -0.255 ,  0.101 ,  0.135 ,  -0.579 ,  0.417 ,  -0.132 ,  0.017 ,  0.040 ,  0.078 ,  0.242 ,  0.314 ,
                                                                                                                                        -0.169 ,  0.005 ,  -0.087 ,  0.341 ,  -0.466 ,  -0.298 ,  -0.222 ,  -0.118 ,  -0.347 ,  0.277 ,  -0.329 ,  -0.227 ,  -0.035 ,  -0.119 ,  0.359 ,  0.000 ,  0.169 ,  0.014 ,  -0.421 ,  0.102 ,  0.385 ,  0.191 ,  -0.104 ,  0.185 ,  -0.217 ,  0.050 ,  0.132 ,  0.178 ,  0.253 ,  0.205 ,  0.265 ,  0.010 ,  0.267 , 
                                                                                                                                        0.137 ,  0.019 ,  -0.145 ,  0.033 ,  -0.174 ,  -0.221 ,  0.089 ,  0.047 ,  0.396 ,  0.098 ,  0.026 ,  -0.461 ,  0.036 ,  0.061 ,  0.413 ,  -0.451 ,  -0.111 ,  -0.217 ,  0.123 ,  0.225 ,  0.113 ,  0.088 ,  -0.012 ,  -0.045 ,  -0.069 ,  -0.484 ,  0.855 ,  0.262 ,  0.023 ,  0.374 ,  0.151 ,  0.172 ,  -0.345 ,  
                                                                                                                                        0.140 ,  -0.741 ,  -0.215 ,  -0.330 ,  0.419 ,  0.329 ,  0.166 ,  -0.251 ,  -0.247 ,  0.046 ,  0.242 ,  -0.057 ,  -0.089 ,  -0.194 ,  -0.314 ,  0.202 ,  -0.117 ,  0.068 ,  0.429 ,  -0.211 ,  0.106 ,  -0.011 ,  -0.073 ,  -0.039 ,  -0.090 ,  -0.165 ,  0.169 ,  0.287 ,  0.149 ,  0.041 ,  0.098 ,  0.427 ,  0.279 ,
                                                                                                                                            0.194 ,  -0.320 ,  -0.122 ,  0.231 ,  0.022 ,  0.586 ,  -0.260 ,  0.074 ,  0.149 ,  -0.388 ,  0.081 ,  -0.079 ,  -0.487 ,  -0.545 ,  -0.090 ,  -0.157 ,  -0.224 ,  0.147 ,  0.291 ,  0.384 ,  0.067 ,  -0.312 ,  0.337 ,  -0.241 ,  0.129 ,  0.501 ,  0.123 ,  0.235 ,  -0.255 ,  0.020 ,  0.104 ,  0.223 ,  -0.278 , 
                                                                                                                                                0.021 ,  -0.037 ,  -0.276 ,  -0.038 ,  -0.240 ,  -0.185 ,  0.305 ,  -0.275 ,  -0.501 ,  -0.264 ,  -0.390 ,  0.232 ,  -0.253 ,  -0.112 ,  -0.107 ,  -0.034 ,  0.096 ,  0.247 ,  0.432 ,  0.126 ,  -0.055 ,  0.073 ,  0.214 ,  0.055 ,  0.304 ,  -0.344 ,  0.059 ,  0.288 ,  0.121 ,  0.351 ,  0.233 ,  0.040 ,  -0.189 , 
                                                                                                                                                -0.029 ,  -0.022 ,  0.230 ,  0.063 ,  0.338 ,  -0.333 ,  0.100 ,  0.347 ,  -0.204 ,  -0.141 ,  0.271 ,  -0.104 ,  -0.343 ,  0.564 ,  -0.113 ,  0.451 ,  -0.009 ,  -0.240 ,  0.029 ,  0.182 ,  -0.216 ,  0.014 ,  -0.323 ,  -0.130 ,  0.128 ,  0.074 ,  0.140 ,  -0.042 ,  0.065 ,  -0.110 ,  0.079 ,  -0.045 ,  0.252 ,  -0.011 , 
                                                                                                                                                    -0.468 ,  -0.124 ,  0.172 ,  0.156 ,  0.157 ,  -0.081 ,  0.234 ,  0.265 ,  -0.165 ,  -0.170 ,  -0.121 ,  -0.234 ,  0.011 ,  0.013 ,  0.050 ,  0.150 ,  0.026 ,  -0.054 ,  -0.260 ,  0.343 ,  -0.373 ,  0.206 ,  -0.270 ,  0.164 ,  0.230 ,  0.024 ,  0.068 ,  0.165 ,  -0.075 ,  0.145 ,  0.114 ,  0.016 ,  -0.369 ,  -0.192 ,  0.275 ,  -0.341 ,
                                                                                                                                                        -0.062 ,  0.235 ,  0.318 ,  -0.134 ,  0.049 ,  -0.022 ,  0.191 ,  0.008 ,  0.598 ,  0.128 ,  0.146 ,  -0.146 ,  0.320 ,  -0.072 ,  -0.137 ,  0.083 ,  -0.138 ,  -0.135 ,  0.397 ,  -0.037 ,  0.068 ,  -0.076 ,  0.084 ,  -0.066 ,  -0.187 ,  -0.186 ,  0.062 ,  0.008 ,  0.268 ,  0.147 ,  0.110 ,  0.046 ,  -0.186 ,  -0.310 ,  -0.428 ,  -0.122 ,
                                                                                                                                                            0.197 ,  -0.221 ,  -0.038 ,  -0.251 ,  -0.196 ,  0.254 ,  0.297 ,  0.250 ,  0.104 ,  -0.036 ,  -0.272 ,  -0.226 ,  0.178 ,  -0.225 ,  -0.051 ,  0.062 ,  0.200 ,  0.042 ,  0.053 ,  0.019 ,  -0.100 ,  -0.121 ,  -0.354 ,  0.314 ,  -0.085 ,  0.034 ,  
                                                                                                                    -0.116 ,  0.349 ,  -0.357 ,  -0.031 ,  -0.330 ,  -0.023 ,  0.091 ,  0.104 ,  -0.416 ,  -0.120 ,  ],requires_grad=True).expand(key_shape))
                    
                    
                    #nn.init.uniform_(self.prompt_key, -1, 1)
                    # print(self.prompt_key.shape)
                    # print(self.prompt_key[0])
                    # input()
            else:
                # else use mean of prompt as key
                # only compatible with prompt, not prefix
                prompt_mean = torch.mean(self.prompt, dim=1)
                self.prompt_key = prompt_mean
                # shape: (pool_size, 1, embedding_size)
                #print(self.prompt_key.requires_grad)
        
        self.prompt_timesused = torch.ones(pool_size)
        self.prompt_timesnotused = torch.ones(pool_size)
        self.prompt_weight = torch.ones(pool_size)
        
        #self.prompt_key = self.prompt_key + promt_base.expand(size=(self.prompt_key.shape))
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.max(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, selected_id = None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                #print(torch.mean(x_embed[0,:,0]))
                #print(torch.mean(x_embed[0,:,1]))
                x_embed_mean_std = torch.cat((torch.mean(x_embed, dim=1),torch.std(x_embed, dim=1)),dim=1)
                # shape: (batch_size, 1, embedding_size)
                #print(x_embed_mean)
            # elif self.embedding_key == 'max':
            #     x_embed_mean = torch.max(x_embed, dim=1)[0]
            #     #print(x_embed_mean)
            # elif self.embedding_key == 'mean_max':
            #     x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            # elif self.embedding_key == 'cls':
            #     if cls_features is None:
            #         x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
            #     else:
            #         x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
            
            # for i in range(64):
            #     plt.plot(np.arange(768),x_embed_mean.cpu().numpy()[i])
            # plt.show()
            #x_embed_mean = self.scalor.transform(x_embed_mean.cpu().numpy())
            #print(x_embed_mean.shape)
            # for i in range(64):
            #     plt.plot(np.arange(768),x_embed_mean[i])
            # plt.show()
            #x_embed_mean = torch.from_numpy(x_embed_mean).to(torch.device('cuda', 0))
            # prompt_norm = self.prompt_key
            # x_embed_norm = x_embed_mean
            #prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            # shape: (pool_size, 1, embedding_size)
            #prompt_norm  = torch.ones(size=(w))
            #x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
            # shape: (pool_size, 1, embedding_size)

            #idx = self.cluster.predict(x_embed_mean.astype(float))
            #print(idx)
            # idx = np.repeat(np.argmax(np.bincount(idx)),x_embed_mean.shape[0])
            # # print(idx)
            # # input()

            #idx = torch.from_numpy(idx).to(torch.device('cuda', 0)).long()

            # print(prompt_norm.type())
            # print(x_embed.type())
            # input()

            # print(x_embed_norm.type())
            # print(prompt_norm.type())

            #similarity = torch.matmul(x_embed_mean_std, self.prompt_key.t()) # B, Pool_size

            x_embed_mean_std_norm = normalize(x_embed_mean_std.cpu().numpy(),axis=1)
            
            x_embed_mean_std_norm_gpu = torch.from_numpy(x_embed_mean_std_norm).to(torch.device('cuda', 0))

            similarity = -torch.cdist(x_embed_mean_std_norm_gpu, self.prompt_key, p=2.0)
            # print(x_embed_mean_std.shape)
            # print(self.prompt_key.shape)
            # print(similarity.shape)
            # input()

            #print(similarity[0])

            # if torch.max(self.prompt_timesnotused) > self.key_div_bound:
            #     #idx = torch.argmax(self.prompt_timesnotused).expand(idx.shape)
            #     # print(self.prompt_key[torch.argmax(self.prompt_timesused)].data.shape)
            #     # print(x_embed.shape)
            #     self.prompt_weight = torch.ones(prompt_norm.shape[0])
            #     #print(torch.nonzero(self.prompt_timesnotused>10))
            #     #self.prompt_weight.index_fill(0,torch.nonzero(self.prompt_timesnotused>10),2.0)
            #     self.prompt_weight[self.prompt_timesnotused>self.key_div_bound]=torch.tensor([2.0])
            #     #self.prompt_weight[self.prompt_timesnotused>10] = 2.0

            #similarity = torch.mul(similarity,self.prompt_weight.expand(similarity.shape))
            #print(similarity[0])
            
            if selected_id is None:
                if prompt_mask is None:
                    _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k

                    if self.batchwise_prompt:
                        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                        # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                        # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                        # Unless dimension is specified, this will be flattend if it is not already 1D.
                        if prompt_id.shape[0] < self.pool_size:
                            prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                            id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                        _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                        major_prompt_id = prompt_id[major_idx] # top_k
                        # expand to batch
                        idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
                else:
                    idx = prompt_mask # B, top_k
            else:
                idx = selected_id

            # if torch.max(self.prompt_timesnotused) > 100:
            #     idx = torch.argmax(self.prompt_timesnotused).expand(idx.shape)
            #     self.prompt_key[torch.argmax(self.prompt_timesnotused)].data = self.prompt_key[torch.argmax(self.prompt_timesused)].data
            #     self.prompt[torch.argmax(self.prompt_timesnotused)].data = self.prompt[torch.argmax(self.prompt_timesused)].data

            #print(idx)
            #print(idx)
            # batch_size = idx.shape[0]
            # for i in range(topk):
            #     idx0 = np.repeat(np.argmax(np.bincount(idx[i].to(torch.device('cpu')).numpy().flatten())),batch_size)

            #     idx_np = np.append(idx_np,idx0,axi))

            # idx = torch.from_numpy(idx0).to(torch.device('cuda', 0)).long()
            # #print(idx)
            # idx = torch.reshape(idx,(batch_size,1))
            
            # self.prompt_timesused[idx]    = self.prompt_timesused[idx] + 1
            # self.prompt_timesnotused = self.prompt_timesnotused + 1
            # self.prompt_timesnotused[idx] = 0

            # print("times used : ")
            # print(self.prompt_timesused)
            # print("times not used ： ")
            # print(self.prompt_timesnotused)
            # make sure all the prompts have been used 
            # if torch.max(self.prompt_timesnotused) > 500:
            #     #print(torch.argsort(self.prompt_timesused))
            #     id = torch.argsort(self.prompt_timesused)[-3:]
            #     #print(id)
            #     prompt_key_copy = self.prompt_key[id].mean(dim=0)
            #     #print(self.prompt_timesnotused>100)
            #     self.prompt_key[self.prompt_timesnotused>500].data = prompt_key_copy

            print("idx shape : ",idx.shape)
            #idx.reshape(idx.shape[1],idx.shape[0])
            batched_prompt_raw = self.prompt[idx] #top_k, B, length, C
            #print(batched_prompt_raw.shape)
            batch_size, topk,  length, c = batched_prompt_raw.shape
            print("selected prompt shape : ",batched_prompt_raw.shape)
            #batched_prompt_raw = batched_prompt_raw.expand(batch_size,self.prompt_size,length,c)
            #print(batched_prompt_raw.shape)
            batched_prompt = batched_prompt_raw.reshape(batch_size,topk, self.prompt_size * length, int(c/self.prompt_size)) # B, top_k * length, C
            # print(batched_prompt.shape)
            # input()

            out['prompt_idx'] = idx
            # shape: (pool_size, top_k)

            # Debugging, return sim as well
            #out['prompt_norm'] = prompt_norm
            #out['x_embed_norm'] = x_embed_norm
            #out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            # batched_key_norm = prompt_norm[idx] # B, top_k, C
            # # shape: (batch_size, top_k, embedding_size)
            # out['selected_key'] = batched_key_norm
            # x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            # # shape: (batch_size, embedding_size)
            # sim = batched_key_norm * x_embed_norm # B, top_k, C
            # # shape: (batch_size, top_k, embedding_size)
            # reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
            # shape: (1)

            out['reduce_sim'] = 1.0
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        #self.prompt.requires_grad = True
        #self.prompt[1] = self.prompt[1].detach()
        #print(self.prompt.requires_grad)
        # mask = torch.ones(size=())
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        #out['total_prompt_len'] = batched_prompt.shape[1]
        #out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        #print(batched_prompt.shape)
        out['prompt_embedding'] = batched_prompt.reshape(batch_size,topk,self.prompt_size,6,self.embed_dim)
        #out['prompt_embedding_2'] = batched_prompt[:,:,128:]
        # shape: (batch, top_k * length, embedding)

        return out
    


