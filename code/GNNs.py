import math
import logging
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.nn.pytorch import GATConv, SAGEConv, APPNPConv
from dgl import function as fn

from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from torch.autograd import Variable


def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)

class General_GCN(object):
    def __init__(self, adj_matrix, features, labels, tvt_nids, cuda=0, hidden_size=64, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, log=True, name='debug', save_path='test_result/', assortative=None, n_mlp=1,n_hop = 3, activation='relu',g_type='var', model_name = 'ugnn',save_model=0, o_act = 'relu', gate_bias = False, detach=0, if_clip =0, use_loss=0, test_assort=0, general_c=9, nor_c_input=0, feat_soft=0, feat_normalize=1):

        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = epochs
        self.save_path = save_path
        self.if_clip=if_clip
        self.model_name = model_name
        self.test_assort = test_assort
        self.feat_normalize = feat_normalize
        # create a logger, logs are saved to GNN-[name].log when name is not None
        if log:
            self.logger = self.get_logger(name)
        else:
            # disable logger if wanted
            self.logger = logging.getLogger()
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda}' if cuda>=0 else 'cpu')
        all_vars = locals()
        self.log_parameters(all_vars)
        self.save_model = save_model
        self.use_loss = use_loss
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'sigmoid':
            act= nn.Sigmoid()
        elif activation == 'identity':
            act = nn.Identity()
        else:
            act = nn.ReLU()


        if o_act == 'relu':
            out_act = nn.ReLU()
        elif o_act == 'tanh':
            out_act = nn.Tanh()
        elif o_act == 'sigmoid':
            out_act= nn.Sigmoid()
        elif o_act == 'identity':
            out_act = nn.Identity()
        elif o_act == 'relu_tanh':
            out_act = [nn.ReLU(),nn.Tanh()]

        self.load_data(adj_matrix, features, labels, tvt_nids, assortative)

        if model_name == 'ugnn':
            self.model = UGNN(self.features.size(1),
                               hidden_size,
                               self.n_class,
                               n_mlp,
                               n_hop,
                               act,
                               dropout,
                               general_c,
                               g_type,
                               detach,
                               out_act,
                               gate_bias,
                               if_clip,
                               nor_c_input,
                               feat_soft)



        elif model_name == 'gcn':
            self.model = GCN_model(self.features.size(1),
                                       hidden_size,
                                       self.n_class,
                                       n_layers,
                                       act,
                                       dropout)

        elif model_name == 'gsage':
            self.model = GraphSAGE_model(self.features.size(1),
                                        hidden_size,
                                        self.n_class,
                                        n_layers,
                                        F.relu,
                                        dropout,
                                        aggregator_type='gcn')
        elif model_name == 'gat':
            heads = ([8] * n_layers) + [1]
            self.model = GAT_model(self.features.size(1),
                                hidden_size,
                                self.n_class,
                                n_layers,
                                F.elu,
                                heads,
                                dropout,
                                attn_drop=0.6,
                                negative_slope=0.2)






    def load_data(self, adj, features, labels, tvt_nids, assortative=None):
        """ preprocess data """
        # features (torch.FloatTensor)
        if sp.issparse(features):
            features = torch.FloatTensor(features.toarray())
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        if self.feat_normalize:
            self.features = F.normalize(self.features, p=1, dim=1)

        if isinstance(labels, np.ndarray):
            labels = torch.LongTensor(labels)
        self.labels = labels
        assert len(labels.size()) == 1
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]
        # number of classes
        self.n_class = len(torch.unique(self.labels))
        # adj for training
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj = sp.csr_matrix(adj)
        self.adj = adj
        self.G = DGLGraph(self.adj).to(self.device)
        if isinstance(assortative, torch.FloatTensor):
            assortative = assortative
        else:
            assortative = torch.FloatTensor(assortative)
        self.G.ndata['a'] = assortative.to(self.device)

        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)
            







    def fit(self):
        """ train the model """
        # move data to device
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        model = self.model.to(self.device)
        if self.model_name == 'appnpx':
            optimizer = torch.optim.Adam([{"params":model.mlp_layers[0].parameters(),
                                         'lr': self.lr,
                                         'weight_decay':self.weight_decay},
                                        {'params':model.mlp_layers[1].parameters(),
                                         'lr': self.lr}])
        else:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

        # keep record of the best validation accuracy for early stopping
        best_val_acc = 0.
        best_val_loss = 100
        # train model
        assort = self.G.ndata['a'].view(-1)
        for epoch in range(self.n_epochs):
            if self.model_name == 'sppnp':
                self.G = self.ppr_matrix
            model.train()

            logits = model(self.G, features)
            loss = F.nll_loss(logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # validate with original graph (without dropout)
            self.model.eval()
            with torch.no_grad():
                logits_eval = model(self.G, features)
            val_acc = self.eval_node_cls(logits_eval[self.val_nid], labels[self.val_nid])
            val_loss = F.nll_loss(logits[self.val_nid], labels[self.val_nid])
            if self.use_loss:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    test_acc = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid])

                    self.logger.info('Epoch [{:3}/{}]: loss {:.4f}, val acc {:.4f}, test acc {:.4f}'
                                .format(epoch+1, self.n_epochs, val_loss.item(), val_acc, test_acc))
    #                 torch.save(model.state_dict(),self.save_path+'best_model.pt')
                    if self.save_model:
                        torch.save(model,self.save_path+'best_model.pt')
                else:
                    self.logger.info('Epoch [{:3}/{}]: loss {:.4f}, val acc {:.4f}'
                                .format(epoch+1, self.n_epochs, val_loss.item(), val_acc))
            else:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid])
                    if self.test_assort:
                        thre=0.5
                        # assort = self.G.ndata['a'].view(-1)
#                         print(assort.shape)
                        test_labels = labels[self.test_nid]
                        test_logits = logits_eval[self.test_nid]
                        confir_test = assort[self.test_nid]

                        logits_low = test_logits[confir_test<thre]
                        labels_low = test_labels[confir_test<thre]
                        test_acc_low = self.eval_node_cls(logits_low, labels_low)

                        logits_high = test_logits[confir_test>=thre]
                        labels_high = test_labels[confir_test>=thre]
                        test_acc_high = self.eval_node_cls(logits_high, labels_high)
                        self.logger.info('Epoch [{:3}/{}]: loss {:.4f}, val acc {:.4f}, test acc {:.4f}, test low acc {:.4f}, test high acc {:.4f} '
                                    .format(epoch+1, self.n_epochs, val_loss.item(), val_acc, test_acc, test_acc_low, test_acc_high))

                    else:
                        self.logger.info('Epoch [{:3}/{}]: loss {:.4f}, val acc {:.4f}, test acc {:.4f}'
                                    .format(epoch+1, self.n_epochs, val_loss.item(), val_acc, test_acc))
    #                 torch.save(model.state_dict(),self.save_path+'best_model.pt')
                    if self.save_model:
                        torch.save(model,self.save_path+'best_model.pt')
                else:
                    self.logger.info('Epoch [{:3}/{}]: loss {:.4f}, val acc {:.4f}'
                                .format(epoch+1, self.n_epochs, val_loss.item(), val_acc))

        if self.save_model:
            torch.save(model, self.save_path+'last_model.pt')
        # get final test result without early stop
        with torch.no_grad():
            logits_eval = model(self.G, features)
        test_acc_final = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid])
        # log both results
        self.logger.info('Final test acc with early stop: {:.4f}, without early stop: {:.4f}'
                    .format(test_acc, test_acc_final))
        if self.test_assort:
            return test_acc,test_acc_low, test_acc_high
        else:
            return test_acc

    def log_parameters(self, all_vars):
        """ log all variables in the input dict excluding the following ones """
        del all_vars['self']
        del all_vars['adj_matrix']
        del all_vars['features']
        del all_vars['labels']
        del all_vars['tvt_nids']
        self.logger.info(f'Parameters: {all_vars}')

    @staticmethod
    def eval_node_cls(nc_logits, labels):
        """ evaluate node classification results """
        preds = torch.argmax(nc_logits, dim=1)
        correct = torch.sum(preds == labels)
        acc = correct.item() / len(labels)
        return acc

    @staticmethod
    def get_logger(name):
        """ create a nice logger """
        logger = logging.getLogger(name)
        # clear handlers if they were created in other runs
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        # create console handler add add to logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler add add to logger when name is not None
        if name is not None:
            fh = logging.FileHandler(f'GAug-{name}.log')
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        return logger




class Film(nn.Module):
    def __init__(self,
                in_feats,
                activation=nn.ReLU(),
                dropout = nn.Dropout(0),
                out_act = nn.Identity(),
                gate_bias = False,
                if_clip = False
                ):
        super(Film, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.out_act = out_act
        self.if_clip = if_clip
        out_feats = 1
        self.layer1 = nn.Linear(in_feats,out_feats, bias=gate_bias)

    def forward(self,features):


        if isinstance(self.out_act, list):
            return self.out_act[1](self.out_act[0](self.layer1(features)))
        else:
            if self.if_clip:
                return torch.clamp(self.out_act(self.layer1(features)),0,1)
            else:
                return self.out_act(self.layer1(features))




class GCN_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_hop,
                 activation,
                 dropout):
        super(GCN_model, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_hop - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(n_hidden, n_classes, None, dropout))


    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return F.log_softmax(h, dim=1)





class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'),
                     fn.sum(msg='m', out='h'))
        h = g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h





class GAT_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 heads,
                 dropout,
                 attn_drop,
                 negative_slope):
        super(GAT_model, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(in_feats, n_hidden, heads[0], dropout, attn_drop, negative_slope, False, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GATConv(n_hidden * heads[i], n_hidden, heads[i+1], dropout, attn_drop, negative_slope, False, activation=activation))
        # output layer
        self.layers.append(GATConv(n_hidden * heads[-2], n_classes, heads[-1], dropout, attn_drop, negative_slope, False, activation=None))
        
    def forward(self, g, features):
        h = features
        for l in range(self.n_layers):
            h = self.layers[l](g, h).flatten(1)
        logits = self.layers[-1](g, h).mean(1)
        return F.log_softmax(logits, dim=1)


class GraphSAGE_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE_model, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=0., activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return F.log_softmax(h, dim=1)




class UGNN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_mlp,
                 n_hop,
                 activation,
                 dropout,
                 general_c,
                 g_type,
                 detach,
                 out_act,
                 gate_bias,
                 if_clip,
                 nor_c_input,
                 feat_soft,
                 batch_norm=0):
        super(UGNN, self).__init__()
        print(out_act)
        self.activation = activation
        self.n_hop = n_hop
        self.batch_norm = batch_norm
        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(nn.Dropout(dropout))
        self.mlp_layers.append(nn.Linear(in_feats,n_hidden))
        self.mlp_layers.append(activation)
        self.mlp_layers.append(nn.Dropout(dropout))
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(n_hidden)
        for i in range(n_mlp - 1):
            self.mlp_layers.append(nn.Linear(n_hidden, n_hidden))
            self.mlp_layers.append(activation)
            self.mlp_layers.append(nn.Dropout(dropout))
        # output layer
        self.mlp_layers.append(nn.Linear(n_hidden,n_classes))
        self.conv = UGNNConv(n_hop, n_classes, general_c, g_type, detach, out_act, gate_bias, if_clip,nor_c_input,feat_soft,dropout)

    def mlp(self, features):
        h = features
        for layer in self.mlp_layers:
            h = layer(h)
        return h


    def forward(self,g, features):
        h = self.mlp(features)
        h = self.conv(g,h)


        return F.log_softmax(h, dim=1)


        
        




class UGNNConv(nn.Module):

    def __init__(self,
                 k,
                 n_classes,
                 general_c,
                 g_type,
                 detach,
                 out_act,
                 gate_bias,
                 if_clip,
                 nor_c_input,
                 feat_soft,
                 dropout):
        super(UGNNConv, self).__init__()
        self._k = k
        self.general_c = general_c
        self.detach = detach
        self.g_type = g_type
        self.nor_c_input = nor_c_input
        self.feat_soft = feat_soft
        if self.g_type == 'var':
            self.c_func = Film(n_classes , activation=nn.ReLU(), dropout = nn.Dropout(0), out_act=out_act, gate_bias=gate_bias, if_clip=if_clip)


    def get_var(self, g, f):
        g.ndata['f'] = f
        g.update_all(fn.copy_src(src='f', out='m'),
                     fn.mean(msg='m', out='mean'))

        g.apply_edges(fn.v_sub_u('mean', 'f', 'e'))
        e = g.edata.pop('e')
        e = e*e
        g.edata['e'] = e
        g.update_all(fn.copy_e(e='e', out ='m'),
                     fn.mean(msg='m', out='var'))

        var = g.ndata.pop('var')
        return var


    def get_c_input(self, g,h):
        if self.feat_soft:
            h = F.normalize(h, p=1, dim=1)
        if self.g_type == 'var':
            var = self.get_var(g,h)
            c_input = var

        if self.detach:
            c_input = c_input.cpu().detach().cuda()
        else:
            c_input = c_input
        return c_input


    def get_c(self, graph, feat):
        if self.g_type == 'constant':
            c = self.general_c*torch.ones(graph.number_of_nodes(), 1).to(feat.device)
        if self.g_type == 'var':
            c_input = self.get_c_input(graph, feat)
            if self.nor_c_input:
                c_input = F.normalize(c_input, p=1, dim=1)
            c = self.general_c*self.c_func(c_input).view(graph.number_of_nodes(),-1)
        return c






    def forward(self, graph, feat):
        graph = graph.local_var()

        c_customized = self.get_c(graph, feat)

        graph.ndata['c'] = c_customized
        graph.apply_edges(fn.u_add_v('c','c','e'))

        e = graph.edata['e']
        graph.update_all(fn.copy_e('e','m'),
                            fn.sum('m', 'c_sum'))
        c_sum = graph.ndata['c_sum']
 
        de_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -1).view(graph.number_of_nodes(),-1)

        b = 1/(2+c_sum*de_norm)

        norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp).to(feat.device)

        feat_0 = feat
        for _ in range(self._k):
            feat = feat * norm
            graph.ndata['h'] = feat

            graph.update_all(fn.u_mul_e('h', 'e', 'm'),
                             fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')

            feat = feat * norm
            feat = b * feat + 2*b* feat_0
        return feat