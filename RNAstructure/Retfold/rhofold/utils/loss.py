# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import partial
from typing import Dict, Optional, Tuple

import ml_collections
import numpy as np
import torch
import torch.nn as nn

from torch.distributions.bernoulli import Bernoulli

from rhofold.utils.rigid_utils import *


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

def sigmoid_cross_entropy(logits, labels):
    logits_dtype = logits.dtype
    logits = logits.double()
    labels = labels.double()
    log_p = torch.nn.functional.logsigmoid(logits)
    # log_p = torch.log(torch.sigmoid(logits))
    log_not_p = torch.nn.functional.logsigmoid(-1 * logits)
    # log_not_p = torch.log(torch.sigmoid(-logits))
    loss = (-1. * labels) * log_p - (1. - labels) * log_not_p
    loss = loss.to(dtype=logits_dtype)
    return loss

def dist_loss(logits, labels, mask):
    logits = logits.float()
    with torch.no_grad():
        labels = torch.nn.functional.softmax(labels, dim=1)
        labels = labels.to(logits.device)
    logits = torch.nn.functional.log_softmax(logits, dim=1)
    mask = mask.to(logits.device)
    loss = -1 * torch.sum(
        labels * logits * mask,
    ) / (torch.sum(mask)+1e-4)
    loss = torch.sum(loss*mask)/(torch.sum(mask)+1e-4)
    return loss

def torsion_angle_loss(
    pred,  # [*, N, 6, 2]
    target,  # [*, N, 6, 2]
):
    # [*, N, 6]
    norm = torch.norm(pred, dim=-1,keepdim=True)
    # [*, N, 6, 2]
    pred = pred / (norm + 1e-4)
    loss_torsion = torch.mean((pred-target)**2)
    return loss_torsion

def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )
    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )
    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))
    assert not torch.any(torch.isnan(normed_error)), print(pred_frames.invert(),pred_positions)
    return normed_error

def masked_msa_loss(logits, true_seq, bert_mask, eps=1e-8, **kwargs):
    """
    Computes BERT-style masked MSA loss. Implements subsection 1.9.9.

    Args:
        logits: [*, length of seq, 5] predicted residue distribution
        true_seq: [*, length of seq] true MSA
        bert_mask: [*, length of seq] MSA mask
    Returns:
        Masked MSA loss
    """
    logits = logits.float()
    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_seq, num_classes=5)
    )

    # FP16-friendly averaging. Equivalent to:
    # loss = (
    #     torch.sum(errors * bert_mask, dim=(-1, -2)) /
    #     (eps + torch.sum(bert_mask, dim=(-1, -2)))
    # )
    loss = errors * bert_mask
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = eps + torch.sum(scale * bert_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = torch.mean(loss)

    return loss

class RetFoldLoss(nn.Module):
    def forward(self,dataflow):

        # dist loss
        pred_dist_p = dataflow['outputs']['p']
        pred_dist_c4_ = dataflow['outputs']['c4_']
        pred_dist_n = dataflow['outputs']['n']

        device = pred_dist_p.device

        target_dist_c4_ = dataflow['expert']['c4_'].to(device)
        target_dist_p = dataflow['expert']['p'].to(device)
        target_dist_n = dataflow['expert']['n'].to(device)
        mask_dist_c4_ = dataflow['feats']['dist_mask_c4_']
        mask_dist_p = dataflow['feats']['dist_mask_p']
        mask_dist_n = dataflow['feats']['dist_mask_n']

        loss_dist_p = dist_loss(pred_dist_p,target_dist_p,mask_dist_p)
        loss_dist_c4_ = dist_loss(pred_dist_c4_,target_dist_c4_,mask_dist_c4_)
        loss_dist_n = dist_loss(pred_dist_n,target_dist_n,mask_dist_n)

        loss_dist = loss_dist_p + loss_dist_c4_ + loss_dist_n

        # torsion loss
        for k,v in dataflow['outputs'].items():
            if isinstance(v,torch.Tensor):
                assert not torch.any(torch.isnan(v)), print(k,v)
        loss_tor_ang = torsion_angle_loss(
            dataflow['outputs']['angles'][-1],
            dataflow['expert']['angles'][-1].to(device)
            )
        
        
        # FAPE loss (ONLY main frame)
        pred_pts = dataflow['outputs']['cord_tns_pred'][-1][0].reshape(-1,23,3)
        expert_pts = dataflow['expert']['cord_tns_pred'][-1][0].reshape(-1,23,3).to(device)
        loss_fape = compute_fape(
            pred_frames=Rigid.from_tensor_7(dataflow['outputs']['frames'][-1,0]),
            target_frames=Rigid.from_tensor_7(dataflow['expert']['frames'][-1,0].to(device)),
            frames_mask=dataflow['feats']['res_mask'],
            pred_positions=pred_pts,
            target_positions=expert_pts,
            positions_mask=dataflow['feats']['atom_mask'],
            length_scale=1,
            l1_clamp_distance=10
        ).mean()

        return loss_dist,loss_tor_ang,loss_fape