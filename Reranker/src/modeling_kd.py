import logging

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, TrainingArguments,AutoTokenizer,AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)
import torch.nn.functional as F

class CrossEncoder(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments,
                kd_weight:float ,
                teacher_model_path:str
                student_tokenizer_path:str):
        
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.config = self.hf_model.config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.margin_loss =  nn.MultiMarginLoss(margin=1.0)
        self.kld_loss = nn.KLDivLoss(reduction='batchmean',log_target=True)
        self.kd_weight = kd_weight
        
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(t_path,trust_remote_code=True )
        self.student_tokenizer = AutoTokenizer.from_pretrained(s_path,trust_remote_code=True )
        
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(t_path,trust_remote_code=True).to('cuda')
        
        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.hf_model.gradient_checkpointing_enable(**kwargs)

    def forward(self, batch):
        
        tt = self.student_tokenizer.batch_decode(batch['input_ids'],skip_special_tokens=True)
        t_ids = self.teacher_tokenizer(tt,return_tensors='pt',padding=True,truncation=True).to('cuda')
        
        with torch.no_grad():
            
            t_output = self.teacher_model(
                **t_ids, return_dict=True, output_hidden_states=True
            )
            t_logits = t_output.logits.view(self.train_args.per_device_train_batch_size,self.data_args.train_group_size)
            
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        
        logits = ranker_out.logits
        

        
        if self.training:
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            ) # batch_size x number of [pos+neg]
            # correct class label is present at zero'th index.
            s_logits =  F.log_softmax(scores, dim=1)
            t_logits =  F.log_softmax(t_logits, dim=1)
            
            loss1 = self.kld_loss(s_logits, t_logits)
            loss2 = self.cross_entropy(scores, self.target_label)
            # loss2 = self.margin_loss(scores, self.target_label)
            
            loss = (1-self.kd_weight)*loss1 + self.kd_weight*loss2
            
            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        state_dict = self.hf_model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.hf_model.save_pretrained(output_dir, state_dict=state_dict)
