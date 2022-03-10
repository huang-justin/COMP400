import scipy.io
import urllib.request
import dgl
import math
import numpy as np
import time
from model import *
data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = './tmp/ACM.mat'
#https://ogb.stanford.edu/kddcup2021/mag240m/
urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)

# mat = {
#         ('paper', 'written-by', 'author') : data['PvsA'].toarray(),
#         ('author', 'writing', 'paper') : data['PvsA'].transpose().toarray(),
#         ('paper', 'citing', 'paper') : data['PvsP'].toarray(),
#         ('paper', 'cited', 'paper') : data['PvsP'].transpose().toarray(),
#         ('paper', 'is-about', 'subject') : data['PvsL'].toarray(),
#         ('subject', 'has', 'paper') : data['PvsL'].transpose().toarray(),
#     }

G = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
    })
print(G)

pvc = data['PvsC'].tocsr()
p_selected = pvc.tocoo()
# generate labels
labels = pvc.indices
labels = torch.tensor(labels).long()

# generate train/val/test split
pid = p_selected.row
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:800]).long()
val_idx = torch.tensor(shuffle[800:900]).long()
test_idx = torch.tensor(shuffle[900:]).long()

#device = torch.device("cuda:0")
G.node_dict = {}
G.edge_dict = {}
for ntype in G.ntypes:
    G.node_dict[ntype] = len(G.node_dict)
for etype in G.etypes:
    G.edge_dict[etype] = len(G.edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype] 
    
#     Random initialize input feature
for ntype in G.ntypes:
    emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 400), requires_grad = False)
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data['inp'] = emb
    

model = HGT(G, n_inp=400, n_hid=200, n_out=labels.max().item()+1, n_layers=2, n_heads=4, use_norm = True)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=100, max_lr = 1e-3, pct_start=0.05)

best_val_acc = 0
best_test_acc = 0
train_step = 0

start = time.time()

for epoch in range(100):
    logits = model(G, 'paper')
    # The loss is computed only for labeled nodes.
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    pred = logits.argmax(1).cpu()
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    val_acc   = (pred[val_idx] == labels[val_idx]).float().mean()
    test_acc  = (pred[test_idx] == labels[test_idx]).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_step += 1
    scheduler.step(train_step)

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    
    if epoch % 5 == 0:
        print('LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
            optimizer.param_groups[0]['lr'], 
            loss.item(),
            train_acc.item(),
            val_acc.item(),
            best_val_acc.item(),
            test_acc.item(),
            best_test_acc.item(),
        ))

end = time.time()
print("training time: ", end-start)