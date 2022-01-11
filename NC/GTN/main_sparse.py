
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model_sparse import GTN, history_GTN
from matplotlib import pyplot as plt
import pdb
from torch_geometric.utils import dense_to_sparse, f1_score, accuracy
from torch_geometric.data import Data
import torch_sparse
import pickle
#from mem import mem_report
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import argparse

import os

import optuna
from optuna.trial import TrialState
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,required=True,
                    help='Dataset')
parser.add_argument('--GNN', type=str,required=True,
                    help='Dataset')
parser.add_argument('--gpu', type=str, default="0",
                    help='gpu=0 1 2 3...')
parser.add_argument('--runs', type=int, default=10,
                    help='number of repetition')
parser.add_argument('--norm', type=str, default='true',
                    help='normalization')
parser.add_argument('--trial_num', type=int, default=1,
                help='the number of trials for each study')
parser.add_argument('--study_name', type=str)
parser.add_argument('--study_storage', type=str)
parser.add_argument('--adaptive_lr', type=str, default='false',
                help='adaptive learning rate')
parser.add_argument('--specify', action="store_true",
                help='no search')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
              
parser.add_argument('--epoch', type=int, default=100,
                    help='Training Epochs')
parser.add_argument('--node_dim', type=int, default=64,
                    help='Node dimension')
parser.add_argument('--num_channels', type=int, default=2,
                    help='number of channels')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='l2 reg')
parser.add_argument('--num_layers', type=int, default=3,
                    help='number of layer')

args = parser.parse_args()


#print(args)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

torch.cuda.set_device(int(args.gpu))
device=torch.device(f"cuda:{int(args.gpu)}")

if args.GNN=="GTN":
    GNN=GTN
elif args.GNN=="history_GTN":
    GNN=history_GTN
else:
    raise Exception


class prio_tuple(tuple):      # tuples with ability to be compared. The first item will be compared.
    def __init__(self,t):
        super(prio_tuple,self).__init__()
    def __gt__(self,other):
        if self[0]>other[0]:
            return True
        else:
            return False
    def __eq__(self,other):
        if self[0]==other[0]:
            return True
        else:
            return False
    def __gq__(self,other):
        if self[0]==other[0] or self[0]>other[0]:
            return True
        else:
            return False
    def __float__(self):
        return float(self[0])






class objective_meta():
    def __init__(self):
        
        self.args=args
        self.runs=args.runs

        self.norm = args.norm
        self.dataset=args.dataset
        self.adaptive_lr=args.adaptive_lr
        self.num_layers =args.num_layers
        self.epoch =args.epoch
        self.dropout=args.dropout
        


    def run_with_trial(self,trial):
        #self.epoch = trial.suggest_int("epoch", 60)
        self.node_dim = trial.suggest_int("node_dim", 32, 128,log=True)
        self.num_channels = trial.suggest_int("num_channels", 1, 4)
        self.lr =  trial.suggest_float("lr", 5e-5, 5e-3,log=True)
        self.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3,log=True)
        #self.num_layers = trial.suggest_int("num_layers", 2, 6)
        #self.adaptive_lr = trial.suggest_categorical("adaptive_lr", ["true", "false"])
        return self.run(trial=trial)[0] # get validation result

    def run_with_specified_params(self,params_dict):
        #self.epoch = params_dict["epoch"]
        self.node_dim = params_dict["node_dim"]
        self.num_channels =params_dict["num_channels"]
        self.lr =  params_dict["lr"]
        self.weight_decay =params_dict["weight_decay"]
        #self.num_layers =params_dict["num_layers"]
        #self.adaptive_lr =params_dict["adaptive_lr"]
        return self.run()[1] # get test result, no trial needed

    def run(self,trial=None):
        #self.get_args(trial)
        runs=self.runs
        norm=self.norm 
        dataset=self.dataset
        adaptive_lr = self.adaptive_lr

        epoch = self.epoch
        node_dim = self.node_dim
        num_channels = self.num_channels
        lr =  self.lr
        weight_decay = self.weight_decay
        num_layers = self.num_layers
        dropout=self.dropout

        with open('data/'+dataset+'/node_features.pkl','rb') as f:
            node_features = pickle.load(f)
        with open('data/'+dataset+'/edges.pkl','rb') as f:
            edges = pickle.load(f)
        with open('data/'+dataset+'/labels.pkl','rb') as f:
            labels = pickle.load(f)
            
            
        num_nodes = edges[0].shape[0]
        A = []
        
        for i,edge in enumerate(edges):
            edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.cuda.LongTensor)
            value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
            A.append((edge_tmp,value_tmp))
        edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
        A.append((edge_tmp,value_tmp))

        node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
        train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
        train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)

        valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
        valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
        test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
        test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)


        num_classes = torch.max(train_target).item()+1

        best_train_losses = []
        best_train_f1s = []
        best_val_losses = []
        best_test_losses = []
        best_val_f1s = []
        best_test_f1s = []
        best_final_f1 = 0
        for cnt in range(runs):
            best_val_loss = 10000
            best_test_loss = 10000
            best_train_loss = 10000
            best_train_f1 = 0
            best_val_f1 = 0
            best_test_f1 = 0
            model = GNN(num_edge=len(A),
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_class=num_classes,
                            num_nodes = node_features.shape[0],
                            num_layers= num_layers,
                            dropout=dropout)
            model.cuda()
            if adaptive_lr == 'false':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                param_temp=[]
                specified_lr=100*lr
                for param_name in model.__dict__["_modules"].keys():
                    if param_name=="oneDimConvs":

                        param_temp.append({"params":  model.__dict__["_modules"][param_name].parameters() ,"lr":specified_lr  })
                    else:
                        param_temp.append({"params":  model.__dict__["_modules"][param_name].parameters() ,"lr":lr  })
                optimizer = torch.optim.Adam(param_temp, lr=lr, weight_decay=weight_decay)

                """optimizer = torch.optim.Adam([{'params':model.gcn.parameters()},
                                            {'params':model.linear1.parameters()},
                                            {'params':model.linear2.parameters()},
                                            {"params":model.oneDimConvs.parameters(), "lr":100*lr}
                                            ], lr=lr, weight_decay=weight_decay)"""
            loss = nn.CrossEntropyLoss()
            Ws = []
            for i in range(epoch):
                #print('Epoch: ',i+1) if  args.specify else None
                for param_group in optimizer.param_groups:
                    if param_group['lr'] > 0.005:
                        param_group['lr'] = param_group['lr'] * 0.9
                model.train()
                model.zero_grad()
                loss, y_train, Ws = model(A, node_features, train_node, train_target)
                loss.backward()
                optimizer.step()
                train_f1 = torch.mean(f1_score(torch.argmax(y_train,dim=1), train_target, num_classes=3)).cpu().numpy()
                #print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1)) if  args.specify else None
                #print(Ws[0]) if  args.specify else None
                model.eval()
                # Valid
                with torch.no_grad():
                    val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target)
                    val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=3)).cpu().numpy()
                    #print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))  if  args.specify else None
                    test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
                    test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=3)).cpu().numpy()
                    test_acc = accuracy(torch.argmax(y_test,dim=1), test_target)
                    #print('Test - Loss: {}, Macro_F1: {}, Acc: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1, test_acc))  if  args.specify else None
                    print(f"epoch: {i:4d} train loss: {loss.detach().cpu().numpy():.4f} train F1: {train_f1:.3f} valid F1: {val_f1:.3f} test F1: {test_f1:.3f}") if  args.specify else None
                    if val_f1 > best_val_f1:
                        best_val_loss = val_loss.detach().cpu().numpy()
                        best_test_loss = test_loss.detach().cpu().numpy()
                        best_train_loss = loss.detach().cpu().numpy()
                        best_train_f1 = train_f1
                        best_val_f1 = val_f1
                        best_test_f1 = test_f1
                        best_Ws=[W.detach().cpu().numpy() for W in Ws]
                        
                

                torch.cuda.empty_cache()
            
            best_test_f1s.append(best_test_f1)
            best_test_f1_mean=sum(best_test_f1s)/len(best_test_f1s)
            best_val_f1s.append(best_val_f1)
            best_val_f1_mean=sum(best_val_f1s)/len(best_val_f1s)

            #score=prio_tuple((best_val_f1_mean,best_test_f1_mean))  # only the validation acc will be compared, the test acc here is just for observing
            score=best_val_f1_mean
            if trial is not None:
                trial.report(score, cnt)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            print('---------------Best Results--------------------')  if  args.specify else None
            print('Train - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_train_f1)) if args.specify else None
            print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1)) if  args.specify else None
            print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1)) if  args.specify else None
            for layer,W in enumerate(best_Ws):
                print(f"Best - W: {W} in layer {layer}" )if  args.specify else None
        print(f"Total performance: val F1: {score} test F1: {best_test_f1_mean}")
        return score,best_test_f1_mean



"""
def objective(trial):

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layer')
    parser.add_argument('--runs', type=int, default=10,
                        help='number of repetition')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                    help='adaptive learning rate')

    args = parser.parse_args()
    #print(args)
    runs=args.runs

    norm = args.norm
    dataset=args.dataset


    epoch = trial.suggest_int("epoch", 20, 100,step=20)
    node_dim = trial.suggest_int("node_dim", 32, 128,log=True)
    num_channels = trial.suggest_int("num_channels", 1, 4)
    lr =  trial.suggest_float("lr", 5e-4, 5e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2,log=True)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    adaptive_lr = trial.suggest_categorical("adaptive_lr", ["true", "false"])




    with open('data/'+dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open('data/'+dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open('data/'+dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)
        
        
    num_nodes = edges[0].shape[0]
    A = []
    
    for i,edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        A.append((edge_tmp,value_tmp))
    edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
    A.append((edge_tmp,value_tmp))

    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)

    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)


    num_classes = torch.max(train_target).item()+1

    best_train_losses = []
    best_train_f1s = []
    best_val_losses = []
    best_test_losses = []
    best_val_f1s = []
    best_test_f1s = []
    best_final_f1 = 0
    for cnt in range(runs):
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        model = history_GTN(num_edge=len(A),
                        num_channels=num_channels,
                        w_in = node_features.shape[1],
                        w_out = node_dim,
                        num_class=num_classes,
                        num_nodes = node_features.shape[0],
                        num_layers= num_layers)
        model.cuda()
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params':model.gcn.parameters()},
                                        {'params':model.linear1.parameters()},
                                        {'params':model.linear2.parameters()},
                                        {"params":model.oneDimConvs.parameters(), "lr":0.5}
                                        ], lr=0.005, weight_decay=0.001)
        loss = nn.CrossEntropyLoss()
        Ws = []
        for i in range(epoch):
            #print('Epoch: ',i+1)
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            model.train()
            model.zero_grad()
            loss, y_train, _ = model(A, node_features, train_node, train_target)
            loss.backward()
            optimizer.step()
            train_f1 = torch.mean(f1_score(torch.argmax(y_train,dim=1), train_target, num_classes=3)).cpu().numpy()
            #print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1))
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target)
                val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=3)).cpu().numpy()
                #print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
                test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=3)).cpu().numpy()
                test_acc = accuracy(torch.argmax(y_test,dim=1), test_target)
                #print('Test - Loss: {}, Macro_F1: {}, Acc: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1, test_acc))
                if val_f1 > best_val_f1:
                    best_val_loss = val_loss.detach().cpu().numpy()
                    best_test_loss = test_loss.detach().cpu().numpy()
                    best_train_loss = loss.detach().cpu().numpy()
                    best_train_f1 = train_f1
                    best_val_f1 = val_f1
                    best_test_f1 = test_f1
                    
            

            torch.cuda.empty_cache()
        
        best_test_f1s.append(best_test_f1)
        best_test_f1_mean=sum(best_test_f1s)/len(best_test_f1s)
        best_val_f1s.append(best_val_f1)
        best_val_f1_mean=sum(best_val_f1s)/len(best_val_f1s)

        #score=prio_tuple((best_val_f1_mean,best_test_f1_mean))  # only the validation acc will be compared, the test acc here is just for observing
        score=best_val_f1_mean
        #print(score)
        trial.report(score, cnt)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        #print('---------------Best Results--------------------')
        #print('Train - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_train_f1))
        #print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        #print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))

        #trial.report(accuracy, epoch)





    return score

"""







if __name__ == "__main__":
    ob=objective_meta()
    #print(args)
    if not args.specify:
        print("start search---------------")
        study = optuna.create_study(study_name=args.study_name, storage=args.study_storage,direction="maximize",pruner=optuna.pruners.MedianPruner(),load_if_exists=True)
        
        study.optimize(ob.run_with_trial, n_trials=args.trial_num)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial




        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        print("computing the corresponding test acc, please wait===================")
        best_test_acc=ob.run_with_specified_params(trial.params)
        print(f"test acc: {best_test_acc}")
        with open(f"./{args.study_name}.txt","a") as f:
            f.write(f"test acc: {best_test_acc:.5f} \n")
            f.write(f"  Value: {trial.value}" )
            
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
    else:
        print("no search, directly execute following params")
        print(args)
        epoch = args.epoch
        node_dim = args.node_dim
        num_channels = args.num_channels
        lr = args.lr
        weight_decay = args.weight_decay
        num_layers = args.num_layers
        norm = args.norm
        adaptive_lr = args.adaptive_lr
        ob.run_with_specified_params(vars(args))

