import scipy.io
import urllib.request
import dgl
import math
import numpy as np
import time
import copy
from HGTDGL.model import *

def runA(test_subject=None):
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

    if test_subject != None:
        subjectCodes = []
        for s in data['L']:
            subjectCodes.append(s[0][0])
        assert test_subject in subjectCodes, "not a vaid subject"
        subject_index = subjectCodes.index(test_subject)

        subject_papers = torch.tensor([]).to(device)
        subject_papers = subject_papers.type(torch.int64)
        subject_papers = torch.concat([subject_papers, (G.successors(subject_index, etype='has'))])
        subject_papers = torch.unique(subject_papers)
        match = set(subject_papers.tolist()) & set(shuffle[900:])
        test_idx = torch.tensor(np.random.permutation(list(match))).long()
    else:
        test_idx = torch.tensor(shuffle[900:]).long()

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
        

    model = HGT(G, n_inp=400, n_hid=200, n_out=labels.max().item()+1, n_layers=2, n_heads=4, use_norm = True).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=100, max_lr = 1e-3, pct_start=0.05)

    best_val_acc = 0
    best_test_acc = 0
    train_step = 0

    start = time.time()
    best_time = time.time()
    best_model = copy.deepcopy(model)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    for epoch in range(100):
        logits = model(G, 'paper')
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))

        pred = logits.argmax(1).cpu()
        train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
        val_acc   = (pred[val_idx] == labels[val_idx]).float().mean()
        test_acc  = (pred[test_idx] == labels[test_idx]).float().mean()

        train_acc_list.append(train_acc.item())
        val_acc_list.append(val_acc.item())
        test_acc_list.append(test_acc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_time = time.time()
            best_model = copy.deepcopy(model)

        if epoch % 10 == 0:
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
    
    best_model_training_time = best_time-start
    total_training_time = end-start

    return best_model, best_model_training_time, total_training_time, train_acc_list, val_acc_list, test_acc_list