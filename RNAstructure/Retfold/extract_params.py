""" """
import logging
import os
import sys
import subprocess
import numpy as np
import random
import torch
import torch.distributed as dist
from rhofold.data.balstn import BLASTN
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils import get_device, save_ss2ct, timing
from rhofold.relax.relax import AmberRelaxation
from rhofold.utils.alphabet import get_features
from rhofold.utils.loss import *
from rhofold.utils.rigid_utils import *
from torch.nn.parallel import DistributedDataParallel as DDP

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

def hallucination_loss(pred,ref):
   
    # pred/ref/back
    pred_dist_n = pred['n']
    pred_dist_c4_ = pred['c4_']
    pred_dist_p = pred['p']
    pred_ss = pred['ss']
    pred_fm = pred['frames']
    pred_pts = pred['cord_tns_pred'][-1][0].reshape(-1,23,3)

    plddt = pred['plddt'][0][0]


    ref_dist_n = ref['n']
    ref_dist_c4_ = ref['c4_']
    ref_dist_p = ref['p']
    ref_ss = ref['ss']
    ref_fm = ref['frames']
    ref_pts = ref['cord_tns_pred'][-1][0].reshape(-1,23,3)
    
    n_res = pred_fm.shape[2] 

    dist_mask = torch.norm(ref_pts[None,:,2,:] - ref_pts[:,None,2,:],dim=-1)

    # dist_mask[dist_mask > 38] = 0
    dist_mask[dist_mask != 0] = 1
    # error loss

    ref_loss_dist = dist_loss(pred_dist_n,ref_dist_n,dist_mask) + \
                    dist_loss(pred_dist_c4_,ref_dist_c4_,dist_mask) + \
                    dist_loss(pred_dist_p,ref_dist_p,dist_mask)

    ref_loss_ss = ss_loss(pred_ss,ref_ss,dist_mask)
    
    frames_mask = torch.ones(n_res)
    pt_mask = torch.ones(pred_pts[:,:3].shape[:-1])
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

    loss = ref_loss_fape # + ref_loss_dist + ref_loss_ss # - 10 * torch.mean(plddt)

    return loss


# @torch.no_grad()
def main(config):
    '''
    RhoFold Inference pipeline
    '''
    
    os.makedirs(config.output_dir, exist_ok=True)

    logger = logging.getLogger('RhoFold Inference')
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
    teacher = RhoFold(rhofold_config)
    student_config = rhofold_config
    student_config["model"]["e2eformer_stack"]["no_blocks"] = 16
    student = RhoFold(student_config)

    logger.info(f'    loading {config.ckpt}')
    teacher.load_state_dict(torch.load(config.ckpt, map_location=torch.device('cpu'))['model'],strict=False)
    student.load_state_dict(torch.load('./pretrained/retnet_5.pt', map_location=torch.device('cpu')),strict=False)
    # print(torch.load(config.ckpt,map_location=torch.device('cpu')))

    teacher.eval()
    student.train()

    # Input seq, MSA
    logger.info(f"Input_fas {config.input_fas}")

    if config.single_seq_pred:
        config.input_a3m = config.input_fas
        logger.info(f"Input_a3m is None, the modeling will run using single sequence only (input_fas)")

    elif config.input_a3m is None:
        config.input_a3m = f'{config.output_dir}/seq.a3m'
        databases = [f'{config.database_dpath}/rnacentral.fasta', f'{config.database_dpath}/nt']
        blast = BLASTN(binary_dpath=config.binary_dpath, databases=databases)
        blast.query(config.input_fas, f'{config.output_dir}/seq.a3m', logger)
        logger.info(f"Input_a3m {config.input_a3m}")

    else:
        logger.info(f"Input_a3m {config.input_a3m}")

    with timing('RhoFold Inference', logger=logger):
        # 0. set up distributed device
        # rank = int(os.environ["RANK"])
        # local_rank = int(os.environ["LOCAL_RANK"])
        # torch.cuda.set_device(rank % torch.cuda.device_count())
        # dist.init_process_group(backend="nccl")
        # device = torch.device("cuda", local_rank)
        # print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        config.device = get_device(config.device)

        # config.device = get_device(config.device)
        logger.info(f'    Inference using device {config.device}')
        teacher = teacher.to(config.device)
        student = student.to(config.device)
        # ---hooks---
        teacher_features = []
        student_features = []
        def teacher_hook(module, input, output):
            teacher_features.append([output[0],output[1]])
        def student_hook(module, input, output):
            student_features.append([output[0],output[1]])
        for i in range(4):
            locals()['teacher_hook_{i}'] = teacher.e2eformer.blocks[-1-3*i].register_forward_hook(teacher_hook)
            locals()['student_hook_{i}'] = student.e2eformer.blocks[-1-4*i].register_forward_hook(student_hook)

        data_dict = get_features(config.input_fas, config.input_a3m)
    
        for name, param in student.named_parameters():
            if "e2eformer" not in name:
                param.requires_grad = False

        # Define Optim
        optim = torch.optim.Adam(student.parameters(),lr=0.0001)
        
        # Forward pass
        token_dict = {'A':4,'U':5,'G':6,'C':7}
        rna_fm_token_dict = {'A':4,'C':5,'G':6,'U':7}
        nt = np.array(['A','C','G','U'])
        # seq=data_dict['seq']
        for ep in range(6,20):

            f = open('./database/t.txt','r')
            lines = f.readlines()
            if ep >= 2:
                random.shuffle(lines)
            count = 0
            avg_plddt = 0
            avg_loss = 0
            avg_fape = 0
            line = lines.pop()
            while lines:
                # ---hooks---
                teacher_features = []
                student_features = []
         
                if random.random() <= 0.5 * (20-ep)/20:
                    seq = np.random.choice(nt,random.randint(8,96))
                    flag = 0
                else:
                    line = line.replace('N',random.choice(nt)).replace('T','U').replace('I','C').replace('X','A')
                    seq = np.array(list(line.rstrip('\n')))
                    count += 1
                    flag = 1
                    line = lines.pop()
                
                hallucinated_seq = torch.tensor([[[token_dict[nt] for nt in seq]]])
                hallucinated_rna_fm_seq = torch.tensor([[rna_fm_token_dict[nt] for nt in seq]])
                optim.zero_grad()
                with torch.no_grad():
                    teacher_output = teacher(tokens=hallucinated_seq.to(config.device),
                        rna_fm_tokens=hallucinated_rna_fm_seq.to(config.device),
                        seq=seq,
                        retention=False
                        )
                student_output = student(tokens=hallucinated_seq.to(config.device),
                        rna_fm_tokens=hallucinated_rna_fm_seq.to(config.device),
                        seq=seq,
                        retention=True
                        )
                loss = 0.0
                mseloss = torch.nn.MSELoss()
                for i in range(1):
                    for j in range(4):
                        decay_recycle = 0.95 ** i
                        decay_depth = 0.8 ** (3-j)
                        loss = loss +  decay_recycle * decay_depth * \
                            (mseloss(student_features[i*4+j][0],teacher_features[i*4+j][0]) + mseloss(student_features[i*4+j][1],teacher_features[i*4+j][1]))
                loss = max(1.0, 2 * teacher_output['plddt'][0][0].mean()) * loss # + hallucination_loss(teacher_output,student_output)
                loss.backward()
                if flag:
                    avg_plddt += teacher_output['plddt'][0][0].mean().cpu().detach().item()
                    avg_loss += loss.cpu().detach().item()
                    avg_fape += hallucination_loss(teacher_output,student_output).cpu().detach()
                optim.step()
                if count % 10 == 1 and flag:
                    logger.info(f"|EPOCH:{ep}|STEP:{count}| Loss {avg_loss/count} pLDDT:{avg_plddt/count} length:{len(seq)} structure loss {avg_fape/count}")
            torch.save(student.state_dict(),f'./pretrained/retnet_{ep}.pt')

def setup_distributed(backend="nccl", port=None):
    """Initialize slurm distributed training environment. (from mmcv)"""
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        os.environ["MASTER_PORT"] = "29500"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)

    dist.init_process_group(backend=backend)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", help="Default cpu. If GPUs are available, you can set --device cuda:<GPU_index> for faster prediction.", default='cpu')
    parser.add_argument("--ckpt", help="Path to the pretrained model, default ./pretrained/model.pt", default='./pretrained/model.pt')
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

    args = parser.parse_args()

    main(args)
