import sys
sys.path.append('../../')
import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import myGAT
import dgl

feature_usage_dict={0:"loaded features",
1:"only target node features (zero vec for others)",
2:"only target node features (id vec for others)",
3:"all id vec. Default is 2",
4:"only term features (id vec for others)",
5:"only term features (zero vec for others)",
}




def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_model_DBLP(args):
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
    exp_info=f"dataset information :\n\tnode num: {adjM.shape[0]}\n\t\tattribute num: {features_list[0].shape[1]}\n\t\tnode type_num: {len(features_list)}\n\t\tnode type dist: {dl.nodes['count']}"+\
                   f"\n\tedge num: {adjM.nnz}"+\
                   f"\n\tclass num: {max(labels)+1}"+\
                   f"\n\tlabel num: {len(train_val_test_idx['train_idx'])+len(train_val_test_idx['val_idx'])+len(train_val_test_idx['test_idx'])} \n\t\ttrain labels num: {len(train_val_test_idx['train_idx'])}\n\t\tval labels num: {len(train_val_test_idx['val_idx'])}\n\t\ttest labels num: {len(train_val_test_idx['test_idx'])}"+"\n"+f"feature usage: {feature_usage_dict[args.feats_type]}"+"\n"+f"exp setting: {vars(args)}"
    print(exp_info)

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])
    

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    
    ma_F1s=[]
    mi_F1s=[]
    for re in range(args.repeat):
        t_re0=time.time()
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_{}_re_{}_feat_{}.pt'.format(args.dataset, args.num_layers,re,args.feats_type))
        for epoch in range(args.epoch):
            t_0_start = time.time()
            # training
            net.train()

            logits = net(features_list, e_feat)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_0_end = time.time()

            # 
            #print('Epoch {:05d} '.format(epoch, )

            t_1_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                logits = net(features_list, e_feat)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            t_1_end = time.time()
            # print validation info
            print('Epoch {:05d} | Train_Loss: {:.4f} | train Time: {:.4f} | Val_Loss {:.4f} | train Time(s) {:.4f}'.format(
                epoch, train_loss.item(), t_0_end-t_0_start,val_loss.item(), t_1_end - t_1_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}_re_{}_feat_{}.pt'.format(args.dataset, args.num_layers,re,args.feats_type)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits = net(features_list, e_feat)
            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_path=f"{args.dataset}_{args.run}.txt")
            pred = onehot[pred]
            d=dl.evaluate(pred)
            print(d)
        ma_F1s.append(d["macro-f1"])
        mi_F1s.append(d["micro-f1"])
        t_re1=time.time();t_re=t_re1-t_re0
        print(f" this round cost {t_re}(s)")
    print(f"mean and std of macro-f1: {  100*np.mean(np.array(ma_F1s)) :.1f}\u00B1{  100*np.std(np.array(ma_F1s)) :.1f}")
    print(f"mean and std of micro-f1: {  100*np.mean(np.array(mi_F1s)) :.1f}\u00B1{  100*np.std(np.array(mi_F1s)) :.1f}")
    print(exp_info)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=10, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--gpu', type=str, default="0")

    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    #torch.cuda.set_device(int(args.gpu))
    #device=torch.device(f"cuda:{int(args.gpu)}")
    run_model_DBLP(args)
