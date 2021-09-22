from GNNs import General_GCN
import numpy as np
import torch
# import matplotlib.pyplot as plt
import dgl.function as fn
from dgl import DGLGraph
import dgl
from dgl import data
import random
import os
import pickle
import networkx as nx
import argparse
from utils import read_data, new_read_data


parser = argparse.ArgumentParser(description='single')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--gnn', type=str, default='ugnn')
parser.add_argument('--new', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--g_type', type = str, default = 'constant')
parser.add_argument('--hidden_size', type = int, default = 16)
parser.add_argument('--wd', type = float, default = 5e-6)
parser.add_argument('--dp', type = float, default = 0.8)
parser.add_argument('--n_mlp', type = int, default = 1)
parser.add_argument('--n_hop', type = int, default = 5)
parser.add_argument('--epochs', type = int, default = 400)
parser.add_argument('--save_model', type = int, default = 0)
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--num_seeds', type = int, default = 30)

parser.add_argument('--o_act', type = str, default = 'sigmoid')
parser.add_argument('--gate_bias', type = int, default = 0)
parser.add_argument('--detach', type = int, default = 1)
parser.add_argument('--if_clip', type = int, default = 0)
parser.add_argument('--ts', type = int, default = 1)
parser.add_argument('--ms', type = int, default = 0)
parser.add_argument('--general_c', type = float, default = 9)
parser.add_argument('--nor_c_input', type = int, default = 0)
parser.add_argument('--feat_soft', type = int, default = 0)
parser.add_argument('--feat_normalize', type = int, default = 1)
parser.add_argument('--num_shuffle', type = int, default = 10)
parser.add_argument('--activation', type = str, default = 'relu')




args = parser.parse_args()

def load_obj(file_name):
    with open(file_name,'rb') as f:
        return pickle.load(f)



    
    
    
def get_assortative(adj, labels):
    adj.setdiag(1)
    g = nx.Graph(adj)
    node_2_labels = dict()
    comfir_num = []
    for node in g.nodes():
    #     print(node)
        neis = list(g.neighbors(node))
        nei_labels = labels[neis]
        node_2_labels[node] = nei_labels
        if nei_labels.size(0)>1:
            cf_n = (torch.nonzero(labels[node]==nei_labels).size(0)-1)/(nei_labels.size(0)-1)
            comfir_num.append(cf_n)
        else:
            comfir_num.append(0)
    comfir_num = np.asarray(comfir_num)
    comfir_num_tensor = torch.FloatTensor(comfir_num)
    ass = comfir_num_tensor.unsqueeze(-1)
    return ass



    
def run( data_name, adj, features, labels, tvt_nids ,torch_seed, model_name, hidden_size,wd,dp,n_mlp, n_hop,g_type, epochs, save_model, d=None, dev= None, lr = 0.01,args=None):  
    labels = torch.LongTensor(labels)

    ass = get_assortative(adj, labels)
    if data_name == 'pubmed':
        num_heads = [8,8]
    else:
        num_heads = [8,1]
    out_dir = 'N_original_results/'
    save_dir = out_dir + 'torch_seed_' +str(torch_seed) +'/' + data_name + '/'  + 'lr_' +str(lr) + '_hidden_size_'  + str(hidden_size) + '_wd_' + str(wd) + '_dp_' + str(dp) + '_n_mlp_'  +str(n_mlp) + 'n_hop' +str(n_hop) + '_gt_' +str(g_type) + '_ep_' + str(epochs)  + '/'
        

    model_dir = save_dir + model_name + '/'
    if not os.path.exists(model_dir):
        if save_model:
            os.makedirs(model_dir)

            
    model_dir = save_dir + model_name +'/'
    gcn_model = General_GCN(adj, features, labels, tvt_nids, cuda=0, hidden_size=hidden_size, epochs=epochs, seed=torch_seed, lr=lr, weight_decay=wd, dropout=dp, log=False, assortative=ass, n_mlp=n_mlp,n_hop = n_hop,activation=args.activation,g_type=g_type,model_name=model_name, save_path = model_dir,save_model = save_model,o_act = args.o_act,gate_bias=args.gate_bias, detach=args.detach,if_clip=args.if_clip, test_assort= args.ts, general_c = args.general_c, nor_c_input=args.nor_c_input, feat_soft=args.feat_soft, feat_normalize = args.feat_normalize)
    
    acc = gcn_model.fit()
    return acc
  
    
    

    

    
def main():
    torch_seed = 187
    torch.manual_seed(torch_seed)
    data_name = args.dataset
    hidden_size = args.hidden_size
    lr =args.lr
    new = args.new
    g_type = args.g_type
    wd = args.wd
    dp = args.dp 
    n_mlp = args.n_mlp
    n_hop = args.n_hop
    epochs = args.epochs
    save_model= args.save_model
    num_seeds = args.num_seeds
    multi_splits = args.ms
    if multi_splits:
        accs = []
        accs_low=[]
        accs_high = []
        for i in range(args.num_shuffle):
            adj, features, labels = new_read_data(data_name)
            tvt_nids = load_obj('../new_data/' + data_name + '_tvt_nids_{}.pkl'.format(i))
            model_name = args.gnn
            torch_seeds = list(range(num_seeds))

            for torch_seed in torch_seeds:
                if args.ts:
                    acc, acc_low, acc_high = run(data_name, adj, features, labels, tvt_nids ,torch_seed ,model_name, hidden_size, wd,dp,n_mlp, n_hop,g_type, epochs, save_model, lr=lr, args=args)
                    accs.append(acc)
                    accs_low.append(acc_low)
                    accs_high.append(acc_high)

                else:
                    acc = run(data_name, adj, features, labels, tvt_nids ,torch_seed ,model_name, hidden_size, wd,dp,n_mlp, n_hop,g_type, epochs, save_model, lr=lr, args=args)
                    accs.append(acc)

        if args.ts:
            print(np.asarray(accs).mean(), np.asarray(accs).std())
            print(np.asarray(accs_low).mean(), np.asarray(accs_low).std())
            print(np.asarray(accs_high).mean(), np.asarray(accs_high).std())
        else:
            print(np.asarray(accs).mean(), np.asarray(accs).std())
        
    else:
        if new:
            adj, features, labels = new_read_data(data_name)
            tvt_nids = load_obj('../new_data/' + data_name + '_tvt_nids_{}.pkl'.format(0))
        else:
            adj, features, labels, tvt_nids = read_data(data_name)

        model_name = args.gnn
        torch_seeds = list(range(num_seeds))
        accs = []
        accs_low=[]
        accs_high = []
        for torch_seed in torch_seeds:
            print(torch_seed)
            if args.ts:
                acc, acc_low, acc_high = run(data_name, adj, features, labels, tvt_nids ,torch_seed ,model_name, hidden_size, wd,dp,n_mlp, n_hop,g_type, epochs, save_model, lr=lr, args=args)
                accs.append(acc)
                accs_low.append(acc_low)
                accs_high.append(acc_high)
                print(acc)
                print(acc_low)
                print(acc_high)

            else:
                acc = run(data_name, adj, features, labels, tvt_nids ,torch_seed ,model_name, hidden_size, wd,dp,n_mlp, n_hop,g_type, epochs, save_model, lr=lr, args=args)
                accs.append(acc)
                print(acc)

        if args.ts:
            print(np.asarray(accs).mean(), np.asarray(accs).std())
            print(np.asarray(accs_low).mean(), np.asarray(accs_low).std())
            print(np.asarray(accs_high).mean(), np.asarray(accs_high).std())
        else:
            print(np.asarray(accs).mean(), np.asarray(accs).std())
    
            
    save_dir = 'experiment_results/' + data_name + '/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = model_name + '_ms_' + str(multi_splits) + '_' +str(args.g_type)
    save_path = save_dir + file_name

    save_results = save_path + '_results.txt'

    with open(save_results, 'wb') as f:
        pickle.dump(accs, f)



    with open(save_path, 'a') as f:
        f.write('==============================================================\n')
        for attr, value in args.__dict__.items():
            line = str(attr) + ': ' + str(value) + '\n'
            f.write(line)
        f.write('--------------------------\n')
        if args.ts:
            line = 'overall: mean: {}, std: {}\n'.format(np.asarray(accs).mean(), np.asarray(accs).std()) 
            f.write(line)
            line = 'low: mean: {}, std: {}\n'.format(np.asarray(accs_low).mean(), np.asarray(accs_low).std()) 
            f.write(line)
            line = 'high: mean: {}, std: {}\n'.format(np.asarray(accs_high).mean(), np.asarray(accs_high).std()) 
            f.write(line)
        else:
            line = 'overall: mean: {}, std: {}\n'.format(np.asarray(accs).mean(), np.asarray(accs).std()) 
            f.write(line)
        f.write('==============================================================\n')










            
                



                

                
                
    
    

    
if __name__ == "__main__":
    main()
    