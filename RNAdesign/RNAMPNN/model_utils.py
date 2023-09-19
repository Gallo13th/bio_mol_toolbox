import torch
import torch.nn as nn
from constants import *
from collections import defaultdict
from typing import List,Dict
from rigid_utils import *
import numpy as np
import random
import os
import pdb_reader
import math

MAX_DISTANCE = 20
N_DISTANCE = 32

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [N,N,C] at Neighbor indices [N,K] => Neighbor features [N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, edges.size(-1))
    edge_features = torch.gather(edges, -2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [N,C] at Neighbor indices [N,K] => [N,K,C]
    # Flatten and expand indices per batch [N,K] => [NK] => [NK,C]
    if nodes.dim() == 3:
        nodes.view(nodes.size(0))
    neighbors_flat = neighbor_idx.view(-1)
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand( -1, nodes.size(-1))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 0, neighbors_flat)
    neighbor_features = neighbor_features.view(nodes.size(0),-1,nodes.size(1))
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def RBF(distance:torch.Tensor,n,max_distance):
    shape = list(distance.shape)
    shape.append(n)
    rbf = torch.zeros(shape)
    sigma = max_distance / n
    for i in range(1,n+1):
        miu = i * sigma
        rbf[...,i-1] = torch.exp(-(distance-miu)**2/(2*sigma**2))
    return rbf

def featurizer(
    structure : Dict,
    device : torch.device,
    n_distance = N_DISTANCE,
    max_distance = MAX_DISTANCE
    ) -> Dict :
    '''
    return feature:
        node:
            frames: List of Rigids [L,**Rigids]
            distance: List of torisions [L, 4(C1',C4',O4',N1/N9),N_DISTANCE(RBF)]
            directions: List of directions [L, 4(C1',C4',O4',N1/N9), 3(xyz)]
        edge:
            orientations: Matrix of Rigids [L,L,**Rigids] (R_i^T*R_j)
            distances: Matrix of distances [L,L,4(C1',C4',O4',N1/N9),N_DISTANCE(RBF)]
            directions: Matrix of directions [L,L,4(C1',C4',O4',N1/N9),3(xyz)]
    '''
    
    
    seq = structure['seq']
    xyz = structure['xyz'].to('cuda:0')
    chain = structure['chain']
    seq_length = len(seq)
    eps = 1e-6
    
    feature = {
        'xyz':xyz,
        'seq':seq,
        'chain':chain,
        'chain_idx': [],
        'residue_idx':[],
        # 'node':{
        #     'frames':None,
        #     'distance':None,
        #     #'direction':None,
        # },
        'edge':{
            #'orientation':None,
            'distance':None,
            #'direction':None,
        },
        }
    # index
    chain_id = list(set(list(chain)))
    count = [0 for _ in chain_id]
    chain_idx = []
    resdue_idx = []
    for i in range(seq_length):
        chain_idx.append(chain_id.index(chain[i]))
        resdue_idx.append(count[chain_id.index(chain[i])])
        count[chain_id.index(chain[i])] += 1
    feature['chain_idx'] = torch.tensor(chain_idx).to('cuda:0')
    feature['residue_idx'] = torch.tensor(resdue_idx).to('cuda:0')
    # # node
    # ## frames
    # frames = Rigid.from_3_points(2*xyz[:,1]-xyz[:,0],xyz[:,1],xyz[:,2]).to_tensor_7()
    # feature['node']['frames'] = frames.to(device)
    # ## distance
    # distance = ((xyz[:,:4]-xyz[:,None,4])**2).sum(dim=-1).sqrt()
    # distance = RBF(distance,n_distance,max_distance)
    # feature['node']['distance'] = distance.to(device)
    ## directions
    # directions = torch.zeros(seq_length,4,3)
    # for i in range(seq_length):
    #     directions[i] = Rigid.from_tensor_7(frames[i]).invert_apply(xyz[i,:4]-xyz[i,None,4])
    # feature['node']['direction'] = directions.to(device)
    
    # edge
    ## orientations
    # orientations = torch.zeros(seq_length,seq_length,7)
    # for i in range(seq_length):
    #     for j in range(seq_length):
    #         orientations[i][j] = Rigid.from_tensor_7(frames[i]).invert().compose(Rigid.from_tensor_7(frames[j])).to_tensor_7()
    # feature['edge']['orientation'] = orientations.to(device)
    ## distances
    distance = ((xyz[:,None,:,None]-xyz[None,:,None,:])**2).sum(dim=-1).sqrt()
    distance = torch.flatten(distance,start_dim=-2,end_dim=-1)
    # print(distance.shape)
    distance = RBF(distance,n_distance,max_distance)
    feature['edge']['distance'] = distance.to(device)
    ## direction
    # directions = torch.zeros(seq_length,seq_length,4,3)
    # for i in range(seq_length):
    #     for j in range(seq_length):
    #         directions[i][j] = Rigid.from_tensor_7(frames[i]).invert_apply(xyz[j,:4]-xyz[i,None,4])
    # feature['edge']['direction'] = directions.to(device)
    return feature

class FeedBackForward(nn.Module):
    
    def __init__(self, num_in, num_ff,num_out) -> None:
        super(FeedBackForward,self).__init__()
        self.W_in = nn.Linear(num_in,num_ff)
        self.W_out = nn.Linear(num_ff,num_out)
        self.act = nn.GELU()
        
    def forward(self,x):
        return self.W_out(self.act(self.W_in(x)))
        
class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E

class RNAFeatures(nn.Module):
    
    def __init__(self,node_features,edge_features,num_position_embedding=16,topk=30,num_rbf=N_DISTANCE,**kargs) -> None:
        super(RNAFeatures,self).__init__()
        self.v_dim = node_features
        self.e_dim = edge_features
        self.pe_dim = num_position_embedding
        self.topk = topk
        
        
        self.embeddings = PositionalEncodings(num_position_embedding)
        edge_in = num_position_embedding + 25 * num_rbf
        # node_in = num_position_embedding + 25 * num_rbf
        # self.node_embedding = nn.Linear(node_in, node_features, bias=False)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        # self.norm_nodes = nn.LayerNorm(node_features)
    
    def _dist(self, xyz, mask, eps=1e-6):
        '''
        mask: 1 is visable, 0 is masked
        xyz [L,3]
        mask [L]
        '''
        mask_2d = mask.unsqueeze(-1)*mask.unsqueeze(-2)
        d = xyz[None,:,:]-xyz[:,None,:]
        d = ((d**2).sum(dim=-1)+eps).sqrt() * mask_2d
        d_max,_ = torch.max(d,-1,keepdim=True)
        d_adj = d + (1. - mask_2d) * d_max
        d_neighbor, e_idx = torch.topk(d_adj, np.minimum(self.topk,xyz.shape[0]),dim=-1,largest=False)
        return d_neighbor, e_idx
    
    def forward(self, structure, device, mask):
        fea = featurizer(structure,device)
        residue_idx = fea['residue_idx']
        chain_idx = fea['chain_idx']
        d_neighbor,e_idx = self._dist(fea['xyz'][:,0],mask)
        seq_length = len(fea['seq'])
        # tmp = []
        # for key in fea['node'].keys():
        #     tmp.append(fea['node'][key].view(seq_length,-1))
        # fea['node'] = torch.concat(tuple(tmp),dim=-1)
        tmp = []
        for key in fea['edge'].keys():
            tmp.append(fea['edge'][key].view(seq_length,seq_length,-1))
        fea['edge'] = torch.concat(tuple(tmp),dim=-1)
        
        #fea['node'] = gather_nodes(fea['node'],e_idx)
        fea['edge'] = gather_edges(fea['edge'],e_idx)
        
        offset = residue_idx[:,None] - residue_idx[None,:]
        offset = gather_edges(offset[...,None],e_idx)[...,0]
        
        d_chains = ((chain_idx[:,None] - chain_idx[None,:])==0).long()
        E_chains = gather_edges(d_chains[...,None],e_idx)[...,0]
        E_position = self.embeddings(offset.long(),E_chains)
        
        fea['edge'] = self.norm_edges(self.edge_embedding(torch.cat((E_position,fea['edge']),-1)))
        #fea['node'] = self.norm_nodes(self.node_embedding(torch.cat((E_position,fea['node']),-1)))        
        
        return fea,d_neighbor,e_idx

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        self.WQ1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.WK1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.WV1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)

        self.WQ2 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.WK2 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.WV2 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None, attention=False):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        if attention:
            query = self.WQ1(h_EV)
            key = self.WK1(h_EV)
            value = self.WV1(h_EV)
            h_message = torch.softmax(query@key.transpose(-1,-2)/math.sqrt(self.num_hidden),dim=-1) @ value
            h_message = self.act(h_message)
        else:
            h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        if attention:
            query = self.WQ2(h_EV)
            key = self.WK2(h_EV)
            value = self.WV2(h_EV)
            h_message = torch.softmax(query@key.transpose(-1,-2)/math.sqrt(self.num_hidden),dim=-1) @ value
            h_message = self.act(h_message)
        else:
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
    
        self.WQ = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.WK = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.WV = nn.Linear(num_hidden + num_in, num_hidden, bias=True)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None, attention=False):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        if attention:
            query = self.WQ(h_EV)
            key = self.WK(h_EV)
            value = self.WV(h_EV)
            h_message = torch.softmax(query@key.transpose(-1,-2)/math.sqrt(self.num_hidden),dim=-1) @ value
            h_message = self.act(h_message)
        else:
            h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

class MPNN(nn.Module):
    
    def __init__(self,node_features,edge_features,num_hidden,num_enc,num_dec,**kargs) -> None:
        super(MPNN,self).__init__()
        self.featurer = RNAFeatures(node_features,edge_features,**kargs)
        self.encoder = nn.ModuleList()
        for _ in range(num_enc):
            self.encoder.append(EncLayer(num_hidden,num_hidden*2,**kargs))
        self.decoder = nn.ModuleList()
        for _ in range(num_dec):
            self.decoder.append(DecLayer(num_hidden,num_hidden*3,**kargs))
        self.predcit_head = nn.Sequential(
            nn.Linear(node_features,4),
            nn.Softmax()
        )

        self.W_e = nn.Linear(edge_features, num_hidden, bias=True)
        self.W_s = nn.Embedding(4, num_hidden)

    def forward(self,structure,mask,chain_M,device='cpu'):
        fea,_,e_idx = self.featurer(structure,device,mask)
        E = fea['edge']
        h_v = torch.zeros((E.shape[0], E.shape[-1]), device=E.device)
        h_e = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  e_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for enc in self.encoder:
            h_v,h_e = enc(h_v,h_e,e_idx,attention=True)
            h_v_attend = h_v
         # Concatenate sequence embeddings for autoregressive decoder
        seq = fea['seq']
        S = seq.replace('A','0').replace('C','1').replace('G','2').replace('U','3')
        S = list(map(int,list(S)))
        S = torch.tensor(S).to(device)
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_e, e_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_e, e_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_v, h_EX_encoder, e_idx)

        chain_M = chain_M*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = e_idx.shape[0]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, iq, jp->qp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 1, e_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder:
            h_ESV = cat_neighbors_nodes(h_v, h_ES, e_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_v = layer(h_v, h_ESV, mask, attention=True)

        probs = self.predcit_head(h_v)
        return probs


# fea,e_idx = RNAFeatures(128,128)(TEST_STRUCTURE,'cpu',torch.ones(50))

# print(fea['node'].shape)
# print(fea['edge'].shape)
# print(fea['chain_idx'])
# print(fea['residue_idx'])

TEST_SEQ = 'A' * 20 + 'U' * 30
TEST_CHIAN = 'A' * 20 + 'B' * 30
TEST_COORD = torch.randn(50,5,3)
TEST_STRUCTURE = {
    'seq':TEST_SEQ,
    'xyz':TEST_COORD,
    'chain':TEST_CHIAN
}

def fake_data(length) -> Dict:
    res = torch.tensor([[
        [1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,0.0,1.0],
        [0.0,0.0,0.0],
        [-1.0,0.0,0.0]
        ]])
    nt = ['A','C','G','U']
    seq = np.random.choice(nt,length)
    seq_one_hot = []
    for i in range(length):
        vec = res[-1].detach().unsqueeze(0)
        if seq[i] == 'A':
            vec = vec + 1
            seq_one_hot.append([1,0,0,0])
        elif seq[i] == 'C':
            vec = vec + 2
            seq_one_hot.append([0,1,0,0])
        elif seq[i] == 'G':
            vec = vec + 3
            seq_one_hot.append([0,0,1,0])
        else:
            vec = vec + 4
            seq_one_hot.append([0,0,0,1])
        res = torch.cat((res,vec),dim=0)
    return res[1:],torch.tensor(seq_one_hot),''.join(seq)

def seq_to_one_hot(seq) -> List:
    res = []
    nts = ['A','C','G','U']
    for nt in seq:
        one_hot = [0,0,0,0]
        one_hot[nts.index(nt)] = 1
        res.append(one_hot)
    return res

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )

if __name__ == '__main__':
    print('===begin===')
    model = MPNN(128,128,128,3,3)
    loss_func = nn.CrossEntropyLoss()
    model.train()
    model.to('cuda:0')
    optim = get_std_opt(model.parameters(), 128, 0)
    # coord,target,seq = fake_data(16)
    # train_dict = {
    #     'seq':seq,
    #     'xyz':coord,
    #     'chain':'A' * 16
    #     }
    # l = 256
    for ep in range(1,100+1):
        # train
        temp = 0.3 + 0.7 * ep/100
        record = open('./recoord.txt')
        line = record.readlines()
        random.shuffle(line)
        split_rate = 0.7
        split_idx = int(split_rate*len(line))
        train_set,valid_set = line[:split_idx],line[split_idx:]
        train_loss = 0
        train_acc = 0
        train_count = 0
        for line in train_set:
            # name,length,seq = line.rstrip('\n').split('\t')
            name,length,seq = line.rstrip('\n').split('\t')
            coord = torch.load(f'./MPNNSet/{name}/coord.pt').to('cuda:0')
            try:
                target = torch.tensor(seq_to_one_hot(seq))
            except:
                continue
            target = target.to('cuda:0')
            train_dict = {
                'seq':seq,
                'xyz':coord,
                'chain':'A' * len(seq)
                }
            mask = (np.random.randn(len(seq)) >= temp).astype(int)
            mask = torch.from_numpy(mask).to('cuda:0')
            h_v = model(train_dict,mask,mask,device='cuda:0')
            loss = loss_func(h_v[mask,:],target[mask,:].float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print(torch.argmax(h_v,dim=-1))
            acc = (torch.argmax(h_v[mask,:],dim=-1) == torch.argmax(target[mask,:],dim=-1)).float().mean()
            train_loss += loss.item()
            train_acc += acc.item()
            train_count += 1
        # valid
        model.eval()
        valid_loss = 0
        valid_acc = 0
        valid_count = 0
        for line in valid_set:
            # name,length,seq = line.rstrip('\n').split('\t')
            name,length,seq = line.rstrip('\n').split('\t')
            coord = torch.load(f'./MPNNSet/{name}/coord.pt').to('cuda:0')
            try:
                target = torch.tensor(seq_to_one_hot(seq))
            except:
                continue
            target = target.to('cuda:0')
            train_dict = {
                'seq':seq,
                'xyz':coord,
                'chain':'A' * len(seq)
                }
            mask = (np.random.randn(len(seq)) >= temp).astype(int)
            mask = torch.from_numpy(mask).to('cuda:0')
            with torch.no_grad():
                h_v = model(train_dict,mask,mask,device='cuda:0')
                loss = loss_func(h_v[mask,:],target[mask,:].float())
            # print(torch.argmax(h_v,dim=-1))
            acc = (torch.argmax(h_v[mask,:],dim=-1) == torch.argmax(target[mask,:],dim=-1)).float().mean()
            valid_loss += loss.item()
            valid_acc += acc.item()
            valid_count += 1
        # test
        test_loss = 0
        test_acc = 0
        test_count = 0
        test_files = os.listdir('./db/rfam')
        for test_file in test_files:
            try:
                name = test_file
                coord,seq = pdb_reader.pdbreader(f'./db/rfam/{test_file}')
                length = coord.size(0)
                try:
                    target = torch.tensor(seq_to_one_hot(seq))
                except:
                    continue
                target = target.to('cuda:0')
                train_dict = {
                    'seq':seq,
                    'xyz':coord,
                    'chain':'A' * len(seq)
                    }
                mask = (np.random.randn(len(seq)) >= temp).astype(int)
                mask = torch.from_numpy(mask).to('cuda:0')
                with torch.no_grad():
                    h_v = model(train_dict,mask,mask,device='cuda:0')
                    loss = loss_func(h_v[mask,:],target[mask,:].float())
                # print(torch.argmax(h_v,dim=-1))
                acc = (torch.argmax(h_v[mask,:],dim=-1) == torch.argmax(target[mask,:],dim=-1)).float().mean()
                test_loss += loss.item()
                test_acc += acc.item()
                test_count += 1
            except:
                pass
        print(f'\
EP:{ep}|\
train loss:{train_loss/train_count:.4f}|\
train acc:{train_acc/train_count:.4f}|\
valid loss:{valid_loss/valid_count:.4f}|\
valid acc:{valid_acc/valid_count:.4f}|\
test loss:{test_loss/test_count:.4f}|\
test acc:{test_acc/test_count:.4f}')

    torch.save(model.state_dict(),'./RNAMPNN_attention.pt')