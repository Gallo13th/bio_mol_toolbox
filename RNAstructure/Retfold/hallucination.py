""" """
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from rhofold.data.balstn import BLASTN
from rhofold.rhofold import RhoFold
from rhofold.config import *
from rhofold.utils import get_device, save_ss2ct, timing
from rhofold.relax.relax import AmberRelaxation
from rhofold.utils.alphabet import get_features,RNAAlphabet
import collections
import math
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast as autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler

from Bio.PDB.PDBParser import PDBParser
from rhofold.utils.loss import *
import copy
from rhofold.model.rna_fm.data import Alphabet, get_rna_fm_token, BatchConverter, RawMSA
from rhofold.utils.constants import *

def dist_loss(logits, labels, mask):
    logits = logits.float()
    labels = torch.nn.functional.softmax(labels, dim=1)
    logits = torch.nn.functional.log_softmax(logits, dim=1)
    loss = -1 * torch.sum(labels * logits * mask) / (torch.sum(mask) + 1e-6)
    return loss

def ss_loss(logits, labels, mask):
    logits = logits.float()
    labels = torch.nn.functional.sigmoid(labels)
    logits = torch.nn.functional.sigmoid(logits)
    loss = -1 * torch.sqrt(torch.sum(((labels - logits)**2)*mask)/(torch.sum(mask)+1e-6) + 1e-6)
    return loss

def dist_dl(logits, labels):
    logits = logits.float()
    labels = torch.nn.functional.log_softmax(labels, dim=1)
    logits = torch.nn.functional.softmax(logits, dim=1)
    loss = torch.mean(torch.sum(logits * (torch.log(logits) - labels),dim=1))
    return loss

def ss_dl(logits, labels):
    logits = logits.float()
    labels = torch.nn.functional.sigmoid(labels)
    logits = torch.nn.functional.sigmoid(logits)
    loss = torch.mean((logits * torch.log( logits / (labels+1e-6) + 1e-6)))
    return loss

def hallucination_loss(pred,ref,site_h=None,pdb_ref=False):
    loss = {}
    pred_dist_n = pred['n']
    pred_dist_c4_ = pred['c4_']
    pred_dist_p = pred['p']
    pred_ss = pred['ss']
    pred_fm = pred['frames']
    pred_pts = pred['cord_tns_pred'][-1][0].reshape(-1,23,3)

    plddt = pred['plddt'][0][0]
    dist_site_h = torch.einsum('i,j->ij',site_h,site_h).to(plddt.device)
    # pred/ref/back
    if not pdb_ref:

        ref_dist_n = ref['n']
        ref_dist_c4_ = ref['c4_']
        ref_dist_p = ref['p']
        ref_ss = ref['ss']
        ref_fm = ref['frames']
        ref_pts = ref['cord_tns_pred'][-1][0].reshape(-1,23,3)
        
        n_res = pred_fm.shape[2] 
        if site_h is None:
            site_h = torch.zeros(n_res)

        dist_mask = torch.norm(ref_pts[None,:,2,:] - ref_pts[:,None,2,:],dim=-1)

        dist_mask[dist_mask > 38] = 0
        dist_mask[dist_mask != 0] = 1
        dist_mask = dist_mask * dist_site_h

        # error loss

        ref_loss_dist = dist_loss(pred_dist_n,ref_dist_n,dist_mask) + \
                        dist_loss(pred_dist_c4_,ref_dist_c4_,dist_mask) + \
                        dist_loss(pred_dist_p,ref_dist_p,dist_mask)

        ref_loss_ss = ss_loss(pred_ss,ref_ss,dist_mask)
        
        frames_mask = site_h
        pt_mask = torch.ones(pred_pts[:,:3].shape[:-1])
        pt_mask[site_h==0] = 0
        ref_loss_fape = compute_fape(
                pred_frames=Rigid.from_tensor_7(pred_fm[-1,0]),
                target_frames=Rigid.from_tensor_7(ref_fm[-1,0]),
                frames_mask=frames_mask.to(pred_dist_n.device),
                pred_positions=pred_pts[:,:3],
                target_positions=ref_pts[:,:3],
                positions_mask=pt_mask.to(pred_dist_n.device),
                length_scale=1,
                l1_clamp_distance=10
            )
        
        ref_loss_fape = torch.sum(ref_loss_fape*dist_mask)/(torch.sum(dist_mask) + 1e-6)
        # loss
        loss['fape'] = ref_loss_fape
        loss['plddt'] = torch.mean(plddt)
        loss['ss'] = ref_loss_ss
        loss['dist'] = ref_loss_dist
    else:
        frames_mask = site_h
        pt_mask = torch.ones(pred_pts[:,:3].shape[:-1])
        pt_mask[site_h==0] = 0
        ref_fm = Rigid.to_tensor_7(ref['fm']).to(pred_dist_n.device)
        ref_pts = ref['pts'].to(pred_dist_n.device)

        dist_mask = torch.norm(ref_pts[None,:,2,:] - ref_pts[:,None,2,:],dim=-1)

        dist_mask[dist_mask > 38] = 0
        dist_mask[dist_mask != 0] = 1
        dist_mask = dist_mask * dist_site_h

        ref_loss_fape = compute_fape(
                pred_frames=Rigid.from_tensor_7(pred_fm[-1,0]),
                target_frames=Rigid.from_tensor_7(ref_fm),
                frames_mask=frames_mask.to(pred_dist_n.device),
                pred_positions=pred_pts[:,:3],
                target_positions=ref_pts[:,:3],
                positions_mask=pt_mask.to(pred_dist_n.device),
                length_scale=1,
                l1_clamp_distance=10
            )
        ref_loss_fape = torch.sum(ref_loss_fape*dist_mask)/(torch.sum(dist_mask) + 1e-6)
        loss['fape'] = ref_loss_fape
        loss['plddt'] = torch.mean(plddt)

    return loss

def pdb_backbone_rigid(filepath) -> (Rigid,torch.Tensor):
    p = PDBParser()
    s = p.get_structure('input',filepath)
    bb_coord = torch.zeros(0,23,3)
    for model in s:
        for chain in model:
            for residue in chain:
                res_coord = torch.zeros(23,3)
                for atom in residue:
                    res_coord[ATOM_NAMES_PER_RESD[residue.resname].index(atom.name)] = torch.tensor(atom.coord)
                bb_coord = torch.cat((bb_coord, res_coord.unsqueeze(0)), dim=0)
    return Rigid.from_3_points(2*bb_coord[:,1]-bb_coord[:,2], bb_coord[:,1], bb_coord[:,0]),bb_coord

@torch.no_grad()
def main(config):
    '''
    RhoFold Inference pipeline
    '''
    # Set Random Seed
    np.random.seed(int(config.random_seed))
    random.seed(int(config.random_seed))
    torch.manual_seed(int(config.random_seed))
    torch.cuda.manual_seed(int(config.random_seed))

    logger = logging.getLogger('RhoFold Training')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(f'{config.output_dir}/log.txt', mode='w')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f'Constructing RhoFold')
    # ref_model
    student_config = rhofold_config
    student_config["model"]["e2eformer_stack"]["no_blocks"] = 16
    reference = RhoFold(student_config)
    logger.info(f'    loading {config.reference_ckpt}')
    reference.load_state_dict(torch.load(config.reference_ckpt, map_location=torch.device('cpu'))['model'],strict=False)
    reference.eval()

    with timing('RhoFold Inference', logger=logger):

        config.device = get_device(config.device)
        logger.info(f'    Inference using device {config.device}')
        reference = reference.to(config.device)

        # Parameters initialization
        token_dict = {'A':4,'U':5,'G':6,'C':7}
        rna_fm_token_dict = {'A':4,'C':5,'G':6,'U':7}
        nt = np.array(['A','C','G','U'])
        M = np.linspace(int(6), int(2), 40000)

        if config.de_novo != 0:
            length = int(config.de_novo)
            seq =  np.random.choice(nt,length)
            with open(config.input_fas,'w') as f:
                f.write(f'>denovo\n{"".join(seq)}') 

        config.input_a3m = config.input_fas
        data_dict = get_features(config.input_fas, config.input_a3m)
        
        # get ref. struct
        if config.ref_pdb :
            ref = {}
            ref['fm'],ref['pts'] = pdb_backbone_rigid(config.ref_pdb)
            ref_pdb = True
        else:
            ref = reference(tokens=data_dict['tokens'].to(config.device),
                    rna_fm_tokens=data_dict['rna_fm_tokens'].to(config.device),
                    seq=data_dict['seq'],
                    )
            ref_pdb = False
        
        # hallucination
        raw_seq = data_dict['seq']

        
        if config.no_hallucinate_site is not None:
            no_h_site = list(map(int,config.no_hallucinate_site.split('-')))
            h_site = list(set(range(len(raw_seq)))-set(no_h_site))
            site_h = torch.zeros(len(raw_seq))
            site_h[h_site] = 1
        else:
            no_h_site = None
            h_site = list(set(range(len(raw_seq))))
            site_h = torch.zeros(len(raw_seq))
        if config.ref_pdb:
            site_h = torch.ones_like(site_h)

        msa = []

        seq = np.random.choice(nt,len(raw_seq))
        if no_h_site is not None:
            seq[no_h_site] = np.array(list(raw_seq))[no_h_site]
        
        hallucinated_seq = torch.tensor([[[token_dict[nt] for nt in seq]]])
        hallucinated_rna_fm_seq = torch.tensor([[rna_fm_token_dict[nt] for nt in seq]])

        for i in range(10000):
            T = 0.1 * (np.exp(np.log(0.5) / 2000) ** i)

            if i == 0:
                pred = reference(tokens=hallucinated_seq.to(config.device),
                            rna_fm_tokens=hallucinated_rna_fm_seq.to(config.device),
                            seq=seq,
                            retention=True
                            )
                baseline = reference(tokens=data_dict['tokens'].to(config.device),
                    rna_fm_tokens=data_dict['rna_fm_tokens'].to(config.device),
                    seq=data_dict['seq'],
                    retention=True
                    )       
                loss = hallucination_loss(pred,ref,site_h,ref_pdb)
                baseline_loss = hallucination_loss(baseline,ref,site_h,ref_pdb)
                past_loss = loss
                past_seq = seq
                
                if loss.item() <= config.lowest_score_msa + baseline_loss.item():
                    msa.append((f'hallucination_msa_{i}_{pred["plddt"][0][0].mean().item()}',seq, pred["plddt"][0][0].mean().item()))
                logger.info(f'ACCEPT,step:{i},BASELINE:{baseline_loss.item():.4f}|PASTLOSS:{past_loss.item():.4f}')
                
            else:
                seq = np.array(list(past_seq))
                n_mutations = round(M[i])
                plddt = pred['plddt'][0][0].cpu().numpy()
                mut_site = np.argpartition(plddt, int(len(plddt) * 0.5))[:int(len(plddt) * 0.5)]
                # mutation
                seq[np.random.choice(mut_site,n_mutations)] = np.random.choice(nt,n_mutations)
                seq = ''.join(seq)

                msa = sorted(msa,key=lambda x:x[2])[:max(len(msa),32)]
                if len(msa) > 0:
                    hallucinated_seq = torch.tensor([
                        [[token_dict[nt] for nt in seq]]
                    ])
                else:
                    hallucinated_seq = torch.tensor([[[token_dict[nt] for nt in seq]]])
                hallucinated_rna_fm_seq = torch.tensor([[rna_fm_token_dict[nt] for nt in seq]])
                
                pred = reference(tokens=hallucinated_seq.to(config.device),
                            rna_fm_tokens=hallucinated_rna_fm_seq.to(config.device),
                            seq=seq,
                            retention=True
                            )            
                loss = hallucination_loss(pred,ref,site_h,ref_pdb)

                delta = loss-past_loss
                delta = delta.cpu().numpy()
                if delta < 0 or np.random.uniform(0,1) < np.exp(-delta/T):
                    logger.info(f'ACCEPT,step:{i},BASELINE:{baseline_loss.item():.4f}|PASTLOSS:{past_loss.item():.4f}--->CURRENTLOSS:{loss.item():.4f}')
                    past_loss = loss
                    past_seq = seq
                    if pred['plddt'][0][0].mean() >= float(config.lowest_plddt):
                        msa.append((f'hallucination_msa_{i}_{pred["plddt"][0][0].mean().item():.4f}',seq,pred["plddt"][0][0].mean().item()))
                else:
                    logger.info(f'REJECT,step:{i},BASELINE:{baseline_loss.item():.4f}|PASTLOSS:{past_loss.item():.4f}-/->CURRENTLOSS:{loss.item():.4f}')
                if (not torch.any(pred['plddt'][0][0] <= 0.4)) and pred['plddt'][0][0].mean() >= float(config.lowest_plddt):
                    break
        
        msa = sorted(msa,key=lambda x:x[2])
        with open(f'{config.output_dir}/hallucination_seq.fasta','w') as f:
            f.write('>hallucination\n'+seq)
        with open(f'{config.output_dir}/hallucination_msa.fasta','w') as f:
            f.write(f'>hallucinate_seq\n{past_seq}\n')
            for name,seq,plddt in msa[:max(len(msa),32)]:
                if plddt-msa[0][2] >= -0.1:
                    f.write(f'>{name}\n{seq}\n')
        data_dict = get_features(f'{config.output_dir}/hallucination_seq.fasta',f'{config.output_dir}/hallucination_msa.fasta')   
        output = reference(tokens=data_dict['tokens'].to(config.device),
                        rna_fm_tokens=data_dict['rna_fm_tokens'].to(config.device),
                        seq=seq,
                        retention=True
                        )
        os.makedirs(config.output_dir, exist_ok=True)

        # Secondary structure, .ct format
        ss_prob_map = torch.sigmoid(output['ss'][0, 0]).data.cpu().numpy()
        ss_file = f'{config.output_dir}/ss.ct'
        save_ss2ct(ss_prob_map, data_dict['seq'], ss_file, threshold=0.5)

        # Dist prob map & Secondary structure prob map, .npz format
        npz_file = f'{config.output_dir}/results.npz'
        np.savez_compressed(npz_file,
                            dist_n = torch.softmax(output['n'].squeeze(0), dim=0).data.cpu().numpy(),
                            dist_p = torch.softmax(output['p'].squeeze(0), dim=0).data.cpu().numpy(),
                            dist_c = torch.softmax(output['c4_'].squeeze(0), dim=0).data.cpu().numpy(),
                            ss_prob_map = ss_prob_map,
                            plddt = output['plddt'][0].data.cpu().numpy(),
                            )
        print(output['plddt'][0].data.cpu().numpy())
        # Save the prediction
        unrelaxed_model = f'{config.output_dir}/unrelaxed_model.pdb'
        node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)
        seq = data_dict['seq']
        atom_mask = []
        for s in seq:
            if s is '-':
                atom_mask.append([0 for _ in range(23)])
            else:
                atom_mask.append([1 for _ in range(23)])
        atom_mask = np.array(atom_mask)
        reference.structure_module.converter.export_pdb_file(data_dict['seq'].replace('-','A'),
                                                         node_cords_pred.data.cpu().numpy(),
                                                         atom_masks=atom_mask,
                                                         path=unrelaxed_model, chain_id=None,
                                                         confidence=output['plddt'][0].data.cpu().numpy(),
                                                         logger=logger)

        # Amber relaxation
        if config.relax_steps is not None and config.relax_steps > 0:
            with timing(f'Amber Relaxation : {config.relax_steps} iterations', logger=logger):
                amber_relax = AmberRelaxation(max_iterations=config.relax_steps, logger=logger)
                relaxed_model = f'{config.output_dir}/relaxed_{config.relax_steps}_model.pdb'
                amber_relax.process(unrelaxed_model, relaxed_model)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    # Basic argument
    parser.add_argument("--device", help="Default cpu. If GPUs are available, you can set --device cuda:<GPU_index> for faster prediction.", default='cpu')
    
    parser.add_argument("--main_ckpt", help="Path to the pretrained model, default ./pretrained/model.pt", default='./pretrained/model.pt')
    parser.add_argument("--background_ckpt", help="Path to the pretrained model, default ./pretrained/model.pt", default='./pretrained/model.pt')
    parser.add_argument("--reference_ckpt", help="Path to the pretrained model, default ./pretrained/model.pt", default='./pretrained/model.pt')

    parser.add_argument("--input_fas", help="Path to the input fasta file. Valid nucleic acids in RNA sequence: A, U, G, C", required=True)
    parser.add_argument("--input_a3m", help="Path to the input msa file. Default None."
                                            "If --input_a3m is not given (set to None), MSA will be generated automatically. ", default=None)
    parser.add_argument("--output_dir", help="Path to the output dir. "
                                             "3D prediction is saved in .pdb format. "
                                             "Distogram prediction is saved in .npz format. "
                                             "Secondary structure prediction is save in .ct format. ", required=True)
    parser.add_argument("--relax_steps", help="Num of steps for structure refinement, default 1000.", default = 1000)
    parser.add_argument("--single_seq_pred", help="Default False. If --single_seq_pred is set to True, "
                                                       "the modeling will run using single sequence only (input_fas)", default=False)
    parser.add_argument("--database_dpath", help="Path to the pretrained model, default ./database", default='./database')
    parser.add_argument("--binary_dpath", help="Path to the pretrained model, default ./rhofold/data/bin", default='./rhofold/data/bin')
    
    # Hallucination argument
    parser.add_argument("--no_hallucinate_site",help="Site not used to hallucination, de novo design without this param.", default=None)
    parser.add_argument("--lowest_score_msa",help="lowest score for add seqs into msa inputs", default=0)
    parser.add_argument("--random_seed",help="random_seed", default=0)
    parser.add_argument("--highest_fape",default=None)
    parser.add_argument("--lowest_plddt",default=None)
    parser.add_argument("--de_novo",default=None)
    parser.add_argument("--ref_pdb",default=None)

    args = parser.parse_args()

    main(args)

