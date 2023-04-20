import json
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from models import GCN_

import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import sys
sys.path.append("..")

from classify import read_node_label, evaluate_embeddings_with_split
from label_utils_functions import completely_imbalanced_split_train, get_labeled_nodes_label_attribute
from rect_package.process import read_graph_as_matrix, graph_of_deepwalk, svd_feature
from rect_package import process
from rect_package.rect import RECT 

with open("configs.json") as config_file:
    configs = json.load(config_file)
    dataset_name = configs.get("dataset_name").get("node")
# training params
nb_epochs = 50
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 200
sparse = True; 
nonlinearity =  'prelu' 
def train_strucuture(adj, features, idx_train, label_list):
    features = process.preprocess_features(features)
    
    preserve_G = graph_of_deepwalk(adj) # M 
    preserve_G = torch.from_numpy(preserve_G)

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    nb_nodes, ft_size = tuple(features.shape)
    idx_train = torch.LongTensor(np.array(idx_train).astype(np.int32))
    features = torch.FloatTensor(features)

    model = RECT(ft_size, hid_units, nonlinearity, dropout=drop_prob)
    loss_function = nn.MSELoss(reduction='sum') #reduction='sum' 
   
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()
        embeds, _ = model(features, sp_adj, sparse) # U
        reconstruct_G = torch.mm(embeds, embeds.t()) # min |M-UU'|
        loss = loss_function(preserve_G, reconstruct_G)/nb_nodes 
        print('Loss:', epoch, loss.item())
        loss.backward()
        optimiser.step()
    
    model.eval()
    return model.embed(features, sp_adj, sparse) # return U^{1}

def train_label(adj, features, idx_train, label_list):
    features = svd_feature(features)
    print('svd:', features.shape, type(features)) # n*d=200

    attribute_labels = get_labeled_nodes_label_attribute(idx_train, label_list, features)
    attribute_labels = torch.FloatTensor(attribute_labels)
    nb_nodes, ft_size = tuple(features.shape)

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(np.array(idx_train).astype(np.int32))
    features = torch.FloatTensor(features)

    model = RECT(ft_size, hid_units, nonlinearity, dropout=0.0)
    loss_function = nn.MSELoss(reduction='sum') #
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()
        embeds, preds = model(features, sp_adj, sparse)
        labeled_preds = preds[idx_train, :]
        loss = loss_function(attribute_labels, labeled_preds)/nb_nodes
        print('Loss:', epoch, loss.item())
        loss.backward()
        optimiser.step()
    
    model.eval()
    return model.embed(features, sp_adj, sparse) # return U^{2}

def run_rect(adj, feature_file, idx_train, label_list):
    features = process.read_features(feature_file) 
    print('+++++++ RECT-N Training +++++++')
    embeds_structure = train_strucuture(adj.copy(), features.copy(), idx_train, label_list)
    print('+++++++ RECT-L Training +++++++')
    embeds_label = train_label(adj.copy(), features.copy(), idx_train, label_list)# this is the key
    
    return embeds_structure, embeds_label

def evaluate_rect(embeds, X_train_idx, X_test_idx, Y_train, Y_test, Y_all):
    embeds = F.normalize(embeds, p=2, dim=1)
    U = embeds.data.numpy()
    vectors = {}
    for i, embedding in enumerate(U):
        vectors[str(i)] = embedding
    res = evaluate_embeddings_with_split(vectors, X_train_idx, Y_train, X_test_idx, Y_test, Y_all, testnum=5)
    global check
    check = (float(res[1])+float(res[0]))/2
    global acc
    acc = check
    print(res)

if __name__ == '__main__':
    import os
    global datafile
    datafile='cora'; single_label = True # set the datafile and labeltype
    #datafile='ppi'; single_label = False # set the datafile and labeltype
    # citeseer, cora and wiki are single label; PPI and Blogcatalog are multi lable
    edge_file = os.path.join("./datasets", datafile, "graph.txt")
    label_file = os.path.join("./datasets", datafile, "group.txt") 
    feature_file = os.path.join("./datasets", datafile, "feature.txt")    
    X, Y = read_node_label(label_file)
    G = read_graph_as_matrix(nodeids=X, edge_file=edge_file)
    
    removed_class= ['0', '1']
    label_rate = 0.3
    X_train_idx, X_test_idx, Y_train, Y_test, X_train_cid_idx, Y_train_cid = completely_imbalanced_split_train(X, Y, train_precent=label_rate, removed_class=removed_class, single_label=single_label)
    print('completely-imbalanced train number', len(X_train_cid_idx))

    embeds_structure, embeds_label = run_rect(G, feature_file, X_train_cid_idx, Y_train_cid)
    #embeds_structure, embeds_label = run_rect(G, feature_file, X_train_idx, Y_train) # train with the balanced labels 
    
    embeds_structure = F.normalize(embeds_structure, p=2, dim=1)
    embeds_label = F.normalize(embeds_label, p=2, dim=1)

    print('------ evaluate RECT-N ---------')
    res = evaluate_rect(embeds_structure, X_train_idx, X_test_idx, Y_train, Y_test, Y_all=Y)
    print('------ evaluate RECT-L ---------')
    res = evaluate_rect(embeds_label, X_train_idx, X_test_idx, Y_train, Y_test, Y_all=Y)
    
    embeds = torch.cat((embeds_structure, embeds_label), 1) 
    print('------ evaluate RECT ---------')
    res = evaluate_rect(embeds, X_train_idx, X_test_idx, Y_train, Y_test, Y_all=Y)

model_file = './src/' + dataset_name + '.pt'
dataset = Planetoid(root='./datasets', name=dataset_name, split='public')
# data = dataset.data
# feat_norm = NormalizeFeatures()
# data = feat_norm(data)

gnn = GCN_(in_channels=dataset.num_node_features, hidden_channels=64, num_layers=2,
           out_channels=dataset.num_classes, jk='last')
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01, weight_decay=5e-4)
# gnn.train()

# for epoch in range(200):
#     optimizer.zero_grad()
#     out = gnn(data.x, data.edge_index)
#     loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
# gnn.eval()
# pred = gnn(data.x, data.edge_index)
# pred = pred.argmax(dim=-1)
# correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
# acc = int(correct) / int(data.test_mask.sum())

print(f'Accuracy: {acc:.4f}')

gnn.save(model_file)