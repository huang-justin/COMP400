import numpy as np
import random
import torch

def Triples(labels, paper_list=None):
    # input: list or np array of labels (size N)
    # output: tensor of size N x 3

    # transform (paper, venue) pairs to (paper, venue, 1) triples
    # adds (paper, incorrect venue, 0) triples at 1:1 ratio
    new_labels = torch.tensor([]).int()
    for i in range(int(labels.size(dim=0))):
        venue_list = list(range(14))
        venue = labels[i]
        venue_list.remove(venue)
        incorrect_venue = random.choice(venue_list)

        # filter subjects only if there is a subject list
        if paper_list == None or i in paper_list:
            new_labels = torch.concat([new_labels, torch.tensor([[i,venue,1]])])
            new_labels = torch.concat([new_labels, torch.tensor([[i,incorrect_venue,0]])])

    return new_labels

def labelAllVenues(labels):
    # input: list or np array of labels
    # output: np array of size (# papers, # venues)
    n = len(labels)
    rows = list(range(n))
    m = labels.max().item()+1
    output = np.zeros((n,m))
    output[rows, labels] = 1
    return output


