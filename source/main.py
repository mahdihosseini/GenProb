from parsing_agent import ParsingAgent
import numpy as np
import save
import json
#import tensorflow as tf
import glob
import torch
import sys



def main(benchmark, dataset, hp, new, start):
    if(new):
        file_name = save.get_name(benchmark,dataset,hp)
    else:
        date = "06-02-2021_14-46-15"
        file_name = "outputs/results-"+date+"-"+benchmark+"-"+dataset+"-"+hp+".csv"
    
    counter = start
    agent = ParsingAgent(benchmark, dataset, hp, new, start)

    while (agent.index < len(agent.sspace)):
        #try:
        qualities, datamodel_dep, performance, layer_info = agent.get_model()
        if qualities.shape[0] != 0:
            
            print(str(agent.index)+'/'+str(len(agent.sspace)))

            performance = np.broadcast_to(performance,(qualities.shape[0],performance.shape[0]))
            to_write = np.concatenate((performance, qualities, datamodel_dep, layer_info), axis=1)
            save.write(file_name,to_write)
        #except Exception as err:
            #agent.index += 1
            #print("Skipping meta")
            #print(err)
    del agent


if __name__ == "__main__":
    benchmark = 'LilJon' #from NATSS, NATST, NAS101, NAS201, DEMOGEN, NLP, zenNET, LilJon
    dataset = 'CIFAR10' #For NASTSS and NATST -> ImageNet16-120, cifar10, cifar100
                                #For DEMOGEN -> NIN_CIFAR10, RESNET_CIFAR10, RESNET_CIFAR100
                                #For zenNet  -> CIFAR10, CIFAR100, ImageNet
                                #For GenProb  -> CIFAR10, CIFAR100
    hp = '37'            #For NATST --> 12 and 90 epochs
                         #For NATST --> 12 and 200 epochs
    new = 1              # 1 --> Start a new excel file and parse all model files     0 --> Append to a file and start from model at index start
    start = 0 

    if(benchmark == 'LilJon'):
        for epoch in range(int(sys.argv[1]), int(sys.argv[2])):   #for GenProb specify range of epochs to parse with command line argument
            print(epoch)                                          #arg1 > arg2 and both in [0,69]
            main(benchmark, dataset, str(epoch), new, start)
    else:
        main(benchmark, dataset, hp, new, start)

  
