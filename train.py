#############################################################################################################
##
##  Source code for training. In this source code, there are initialize part, training part, ...
##
#############################################################################################################

import cv2
import torch
import visdom
import agent
import numpy as np
from data_loader import Generator
from parameters import Parameters
import test
import evaluation

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Training():
    print('Training')

    ####################################################################
    ## Hyper parameter
    ####################################################################
    print('Initializing hyper parameter')

    vis = visdom.Visdom()
    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='epoch',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))
    
    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = Generator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
    else:
        lane_agent = agent.Agent()
        lane_agent.load_weights(4235, "tensor(0.2127)")

    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    step = 0
    for epoch in range(p.n_epoch):
        lane_agent.training_mode()
        for inputs, target_lanes, target_h, test_image in loader.Generate():
            #training
            print("epoch : " + str(epoch))
            print("step : " + str(step))
            loss_p = lane_agent.train(inputs, target_lanes, target_h, epoch, lane_agent)
            loss_p = loss_p.cpu().data
            
            if step%50 == 0:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * int(step/50),
                    Y=torch.Tensor([loss_p]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')
                
            if step%100 == 0:
                lane_agent.save_model(int(step/100), loss_p)
                testing(lane_agent, test_image, step, loss_p)
            step += 1


        #evaluation
        if epoch > 0 and epoch%10 == 0:
            print("evaluation")
            lane_agent.evaluate_mode()
            th_list = [0.3, 0.5, 0.7]
            lane_agent.save_model(int(step/100), loss_p)

            for th in th_list:
                print("generate result")
                print(th)
                test.evaluation(loader, lane_agent, thresh = th, name="test_result_"+str(epoch)+"_"+str(th)+".json")

            for th in th_list:
                print("compute score")
                print(th)
                with open("eval_result_"+str(th)+"_.txt", 'a') as make_file:
                    make_file.write( "epoch : " + str(epoch) + " loss : " + str(loss_p.cpu().data) )
                    make_file.write(evaluation.LaneEval.bench_one_submit("test_result_"+str(epoch)+"_"+str(th)+".json", "test_label.json"))
                    make_file.write("\n")

        if int(step)>700000:
            break


def testing(lane_agent, test_image, step, loss):
    lane_agent.evaluate_mode()

    _, _, ti = test.test(lane_agent, np.array([test_image]))

    cv2.imwrite('test_result/result_'+str(step)+'_'+str(loss)+'.png', ti[0])

    lane_agent.training_mode()

    
if __name__ == '__main__':
    Training()

