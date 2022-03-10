from operator import sub
import scipy.io
import urllib.request
import dgl
import math
import numpy as np
import time
from HGTDGL.model import *
data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = './HGTDGL/tmp/ACM.mat'

urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)

device = torch.device("cuda:0")

G = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
    }).to(device)

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
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(device) * G.edge_dict[etype] 
    
#     Random initialize input feature
for ntype in G.ntypes:
    emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 400), requires_grad = False).to(device)
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data['inp'] = emb


# # subgraph based off subject 
subjectCodes = []
for a in data['L']:
    subjectCodes.append(a[0][0])
print(subjectCodes)

# x = torch.tensor([])
# x = x.type(torch.int64)
# x = torch.concat([x,(G.successors(0, etype='has'))])
# subgraph = dgl.sampling.sample_neighbors(G, {'subject':0, 'paper':x}, -1, edge_dir='out')
# ['A.0', 'A.1', 'A.m', 
# 'B.2', 'B.3', 'B.4', 'B.5', 'B.6', 'B.7', 'B.8', 
# 'C.0', 'C.1', 'C.2', 'C.3', 'C.4', 'C.5', 'C.m', 
# 'D.0', 'D.1', 'D.2', 'D.3', 'D.4', 'D.m', 
# 'E.0', 'E.1', 'E.2', 'E.3', 'E.4', 'E.5', 'E.m', 
# 'F.0', 'F.1', 'F.2', 'F.3', 'F.4', 'F.m', 
# 'G.0', 'G.1', 'G.2', 'G.3', 'G.4', 'G.m', 
# 'H.0', 'H.1', 'H.2', 'H.3', 'H.4', 'H.5', 'H.m', 
# 'I.1', 'I.2', 'I.3', 'I.4', 'I.5', 'I.6', 'I.7', 'I.m', 
# 'J.0', 'J.1', 'J.2', 'J.3', 'J.4', 'J.5', 'J.6', 'J.7', 'J.m', 
# 'K.0', 'K.1', 'K.2', 'K.3', 'K.4', 'K.6', 'K.m']

target = 'H.2'
subject = subjectCodes.index(target)
subjects = [subject]

subject_papers = torch.tensor([]).to(device)
subject_papers = subject_papers.type(torch.int64)
for s in subjects:
    subject_papers = torch.concat([subject_papers, (G.successors(s, etype='has'))])
subject_papers = torch.unique(subject_papers)

# get subset for train/val/test 
pid = subject_papers.tolist()
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:800]).long()
val_idx = torch.tensor(shuffle[800:900]).long()
test_idx = torch.tensor(shuffle[900:]).long()


# # do I want this?
# subgraph.add_nodes(G.num_nodes('author')- subgraph.num_nodes('author'), ntype='author')
# subgraph.add_nodes(G.num_nodes('paper') - subgraph.num_nodes('paper'), ntype='paper')
# subgraph.add_nodes(G.num_nodes('subject') - subgraph.num_nodes('subject'), ntype='subject')


# #device = torch.device("cuda:0")
# subgraph.node_dict = {}
# subgraph.edge_dict = {}
# for ntype in subgraph.ntypes:
#     subgraph.node_dict[ntype] = len(subgraph.node_dict)
# for etype in subgraph.etypes:
#     subgraph.edge_dict[etype] = len(subgraph.edge_dict)
#     subgraph.edges[etype].data['id'] = torch.ones(subgraph.number_of_edges(etype), dtype=torch.long) * subgraph.edge_dict[etype] 
    
# #     Random initialize input feature
# for ntype in subgraph.ntypes:
#     emb = nn.Parameter(torch.Tensor(subgraph.number_of_nodes(ntype), 400), requires_grad = False)
#     nn.init.xavier_uniform_(emb)
#     subgraph.nodes[ntype].data['inp'] = emb


model = HGT(G, n_inp=400, n_hid=200, n_out=labels.max().item()+1, n_layers=2, n_heads=4, use_norm = True).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=100, max_lr = 1e-3, pct_start=0.05)

best_val_acc = 0
best_test_acc = 0
train_step = 0

start = time.time()
best_time = time.time()

for epoch in range(100):
    logits = model(G, 'paper')
    # The loss is computed only for labeled nodes.
    loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device)  )

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
        best_time = time.time()
    
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
print('best time:', best_time - start)