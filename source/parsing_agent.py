import collections
import sys
import os
#import tensorflow as tf
from torch.utils import data
import process
import numpy as np
from nats_bench import create
from loadmodels import get_cell_based_tiny_net
import glob
import json
import torch
from qc_nin import nin as NIN


class ParsingAgent:
    index = 0
    api = None
    sspace = None

    def __init__(
            self,
            bench: str,
            dataset: str,
            hp: str,
            new: int,
            start: int):

        self.bench = bench
        self.dataset = dataset
        self.hp = hp

        if(bench == 'NATSS'):
            #download dataset?
            self.api = create(sys.path[0][0:-7]+'/models/NATSS', 'sss', fast_mode=True, verbose=False)
            self.sspace = glob.glob(sys.path[0][0:-7]+'/models/NATSS/*')
            #Compute Canada Directories
            
        # if(bench == 'NATSS'):
        #     self.api = create('/home/damounik/scratch/NATS-sss-v1_0-50262-full', 'sss', fast_mode=True, verbose=False)
        #     self.sspace = glob.glob('/home/damounik/scratch/NATS-sss-v1_0-50262-full/*')

        # if(bench == 'NATST'):
        #     #download dataset?
        #     self.api = create('/home/damounik/projects/def-plato/damounik/QC-Bench/QC-Bench/models/NATST/NATS-tss-v1_0-3ffb9-full', 'tss', fast_mode=True, verbose=False)
        #     self.sspace = glob.glob('/home/damounik/projects/def-plato/damounik/QC-Bench/QC-Bench/models/NATST/NATS-tss-v1_0-3ffb9-full/*')



        if(bench == 'NATST'):
            #download dataset?
            self.api = create(sys.path[0][0:-7]+'/models/NATST', 'tss', fast_mode=True, verbose=False)
            self.sspace = glob.glob(sys.path[0][0:-7]+'/models/NATST/*')
            #Compute Canada Directories
            #~/projects/def-plato/damounik/QC-Bench/QC-Bench/models/NATST/NATS-tss-v1_0-3ffb9-full/
            #
        
        elif(bench == 'DEMOGEN'):
            self.sspace = []
            self.sspace.extend(glob.glob(sys.path[0][0:-7]+"/models/DEMOGEN/ydjiang/experimental_results/model_dataset/"+self.dataset+"/*"))
            #print("Folders: ", self.sspace)


        elif(bench == 'NLP'):
            self.sspace = glob.glob("../models/NAS-Bench-NLP/*")
            #print('Folders: ', self.sspace)

        elif(bench == "zenNET"):
            self.sspace = glob.glob("../models/zenNAS/" + self.dataset + "/*")

        elif(bench == 'LilJon'):
            self.sspace = glob.glob(sys.path[0][0:-7]+"/models/LilJon/raw/*_" + self.dataset + "_epoch_" + self.hp + ".pth.tar")

        if(new != 1):
            self.index = start


    def get_model(self):
        #try:
        if(self.bench[0:-1] == 'NATS'):
            self.api.arch2infos_dict = collections.OrderedDict()
            model_path = self.sspace[self.index]

            model_num = int((model_path.split(os.path.sep)[-1]).split('.')[0])

            weightdict = self.api.get_net_param(model_num, self.dataset, hp=self.hp, seed=None)

            config = self.api.get_net_config(model_num,self.dataset)
            model = get_cell_based_tiny_net(config)
            model.load_state_dict(next(iter(weightdict.values())))
            print(model)
            
            '''for weight in list(model.parameters()):
                print(weight)
                '''
            weights = list((list(weightdict.values())[0]).values())
            weights = [weight for weight in weights if (len(weight.shape)==4)]
            '''
            connecs = []
            for key, value in list(weightdict.values())[0]:
                key = key.split('.')
                if(len(value.shape)==4 and 'layers' in key):
                    connec = 0
                    if(key[3]==0):
                        connec = '0'
                    elif(key[3]==1):
                        connec = '01'
                    connecs.append(key[1]+'.'+)
            '''

            performance = self.api.get_more_info(model_num, self.dataset, hp=self.hp, is_random=False)
            performance = [performance['test-accuracy']/100,performance['test-loss'],performance['train-accuracy']/100,performance['train-loss'],performance['test-accuracy']/100-performance['train-accuracy']/100]

        if(self.bench == 'DEMOGEN'):
            print("demo")
            '''
            with tf.compat.v1.Session() as sess:
                #Get Model Number
                model_num = self.index

                #Load Model
                new_saver = tf.compat.v1.train.import_meta_graph(self.sspace[model_num] + "/model.ckpt-150000.meta")
                new_saver.restore(sess, self.sspace[model_num] + '/model.ckpt-150000')
                
                #Extract Weights
                variables_names  = [v.name for v in tf.compat.v1.trainable_variables()]
                values = sess.run(variables_names)
                weights = []
                for k, v in zip(variables_names, values):
                    print(k)
                    if('conv' in k and "kernel" in k):
                        weights.append(torch.tensor(v.transpose((3,2,0,1))))
            
            #Reset Global Variables
            tf.compat.v1.reset_default_graph()

            #Extract Performance
            with open(self.sspace[self.index] + "/eval.json", "r") as read_file:
                eval_data = json.load(read_file)
            with open(self.sspace[self.index] + "/train.json", "r") as read_file:
                train_data = json.load(read_file)
            
            performance = [eval_data["Accuracy"], eval_data["loss"], train_data["Accuracy"], train_data["loss"], eval_data["Accuracy"] - train_data["Accuracy"]]
            '''
        
        if(self.bench == 'NLP'):
            model_num = self.index
            print("Model: ", model_num)

            #Get Weights
            weights = []
            model = torch.load(self.sspace[self.index])
            for name in model:
                if('raw' in name and len(model[name].shape) == 2):
                    #print(name, model[name].shape)
                    weights.append(model[name])

            #Get Performance
            suffix = self.sspace[self.index].split('\\')[-1].replace("dump_weights_model_", "").replace('.pt',"")
            log = json.load(open('../nas-bench-nlp-release-master/train_logs_single_run/log_stats_model_' + suffix + '.json', 'r'))
            performance = [0, log['test_losses'][-1], 0, log['train_losses'][-1], 0]

        if(self.bench == 'zenNET'):
            model_num = self.index
            print("Model: ", model_num)

            #Get Model
            model = torch.load(self.sspace[self.index])

            #Get Weights
            weights = []
            for key in model["state_dict"].keys():
                if(len(model["state_dict"][key].shape) == 4 and not ("proj" in key)):
                    #print(key, " : ", model["state_dict"][key].shape)
                    weights.append(model["state_dict"][key])

            #Get Performance
            performance = [model['top1_acc'], 0, 0, 0, 0]

        if(self.bench == 'LilJon'):
            model_num = self.sspace[self.index].split("NIN")[1][0:5]
            print("Model: ", model_num)

            #Get Model
            model_info = torch.load(self.sspace[self.index])

            #Get Weights
            weights = []
            for key in model_info["state_dict_network"].keys():
                if(len(model_info["state_dict_network"][key].shape) == 4 and ('onv' in key or 'down' in key)):
                    #print(key, " : ")#, model["state_dict"][key].shape)
                    weights.append(model_info["state_dict_network"][key].cpu())

            #Test_acc, Test_loss, Train_acc, Train_loss, Test_acc - Train_acc
            performance = [model_info["performance_statistics"]['test_acc1_epoch_' + self.hp],  
                           model_info["performance_statistics"]['test_loss_epoch_' + self.hp],  
                           model_info["performance_statistics"]['train_acc1_epoch_' + self.hp], 
                           model_info["performance_statistics"]['train_loss_epoch_' + self.hp], 
                           model_info["performance_statistics"]['test_acc1_epoch_' + self.hp] - model_info["performance_statistics"]['train_acc1_epoch_' + self.hp]]
            
            print("num classes: "+self.dataset.split('R')[-1])
            model = NIN(self.sspace[self.index].split('_')[-9][-5:],int(self.dataset.split('r')[-1]))
            model.load_state_dict(model_info["state_dict_network"])
            #del(model_info)

        #except Exception as error:
        #    print(type(error))
        #    print(error)
        #    return None


        qualities, channel_weights, layer_type = self.process_weights(weights, model)
        id = np.expand_dims(np.broadcast_to(model_num, len(channel_weights)),axis=1)
        layer_type = np.expand_dims(np.broadcast_to(layer_type, len(channel_weights)),axis=1)
        layer_info = np.concatenate((id,layer_type,np.asarray(channel_weights)),axis=1)
        
        #try:
        datamodel_dep = process.get_dataset_dep(model, self.dataset, 0, [0,1000], [[0,0],2])
        #except Exception as exc:
            #print(exc)
            #datamodel_dep = np.zeros(4)

        datamodel_dep = np.broadcast_to(datamodel_dep,(len(channel_weights),len(datamodel_dep)))

        del model

        self.index+=1
        return np.asarray(qualities), datamodel_dep, np.asarray(performance), layer_info

    def process_weights(self, weights, model=None):
        qualities = []
        channel_weights = []

        for weight in weights:
            layer_qualities, layer_weights, layer_type = process.get_metrics(weight)
            qualities.append(layer_qualities)
            channel_weights.append(layer_weights)

        return qualities, channel_weights, layer_type