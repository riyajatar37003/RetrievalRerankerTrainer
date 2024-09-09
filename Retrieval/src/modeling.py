import logging
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kld_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config
        
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
        if normlized:
            if self.temperature > 0.5:
                raise ValueError("Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous(),psg_out.last_hidden_state

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))
        
    def multivector_score(self,qreps,presp):

        qreps = torch.nn.functional.normalize(qreps, dim=-1)
        presp = torch.nn.functional.normalize(presp, dim=-1)
        
        # Step 1: Expand dimensions of q and p for broadcasting
        q_expanded = qreps.unsqueeze(1)  # Shape: (2, 1, 10, 32)
        p_expanded = presp.unsqueeze(0)  # Shape: (1, 16, 152, 32)
        
        # Step 2: Perform matrix multiplication using torch.matmul
        # We want to multiply along the last dimension (size 32)
        result = torch.matmul(q_expanded, p_expanded.transpose(-1, -2))  # Shape: (2, 16, 10, 152)
        
        # The resulting shape should be (2, 16, 10, 152), which represents:
        # 2 batches of queries, each compared with 16 batches of passages, 
        # resulting in 10x152 similarity scores.
        score,_ = torch.max(torch.max(result,axis=-1)[0],axis=-1)
        return score
        

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps,q_h = self.encode(query)
        p_reps,p_h = self.encode(passage)
        mv_scores = self.multivector_score(q_h,p_h)
        
        
        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps)/self.temperature #/ self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)
                mv_scores = mv_scores.view(q_reps.size(0), -1)/self.temperature
                
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size

                
                w1 = 1
                w3 = 0.3
                lambda1 = 1
                lammbda2 = 1
                
                s_inter = w1*scores + w3*mv_scores
                
                loss_dense1 = self.compute_loss(scores, target)
                loss_mulvec1 = self.compute_loss(mv_scores, target)
                loss_inter = self.compute_loss(s_inter, target)
                
                s_logits1 =  F.log_softmax(scores, dim=1)
                s_logits2 =  F.log_softmax(mv_scores, dim=1)
                
                t_logits =  F.log_softmax(s_inter, dim=1)

                loss_dense2 = self.kld_loss(s_logits1, t_logits)
                loss_mulvec1 = self.kld_loss(s_logits2, t_logits)

                L1 = 0.3*(loss_dense1*lambda1 + loss_mulvec1*lammbda2 + loss_inter)
                L2 = 0.2*(loss_dense2*lambda1  + loss_mulvec1*lammbda2)
                
                loss = (L1 + L2)*0.5
                # In order to reduce the impact of this, we set w1 =1,w2 =0.3, w3 = 1,λ1 = 1,λ2 = 0.1and λ3 = 1

                
                
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1)*np.exp(3) #/ self.temperature # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
        # 

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
