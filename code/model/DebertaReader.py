from transformers import DebertaV2PreTrainedModel, DebertaV2Model
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import ipdb

########## DebertaReader ##########
class DebertaReader(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.sentence_outputs = nn.Linear(config.hidden_size, 2)
        self.answer_typeout = nn.Linear(config.hidden_size, 3)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_smoothing_method = None  # ["None", "LabelSmoothing", "WordOverlapping", "F1Smoothing"]
        self.smoothing_weight = 0
        self.epoch = 0

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions=None,
        end_positions=None,
        sentence_index=None,
        sentence_labels=None,
        answer_type=None,
        sentence_num=None,
        label_smoothing_start_label=None,
        label_smoothing_end_label=None,
        word_overlapping_start_label=None,
        word_overlapping_end_label=None,
        F1_smoothing_start_label=None,
        F1_smoothing_end_label=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        sentence_index:The index position of each sentence corresponding to </e>. (B, sentence_num)
        sentece_labels:Whether each sentence is a supporting fact. (B, sentence_num)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        # Supporting Facts
        B, L, E = sequence_output.size()
        S = sentence_index.size(1)
        device = sequence_output.device
        sentence_output = torch.zeros([B, S, E], device=device)
        for b in range(B):
            sentence_output[b] = torch.index_select(sequence_output[b], 0, sentence_index[b])
        sentence_select = self.sentence_outputs(sentence_output)
        # Supporting Facts Loss
        Lsentence = None
        if sentence_labels is not None:
            sentence_loss = CrossEntropyLoss(label_smoothing=self.smoothing_weight)
            sentencemask = torch.tensor([5, -5], device=device)
            for b in range(B):
                sentence_select[b, sentence_num[b] :] = sentencemask
            sentence_select_p = sentence_select.permute(0, 2, 1)
            sentence_labels = sentence_labels[:, : sentence_select_p.size(-1)]
            Lsentence = sentence_loss(sentence_select_p, sentence_labels)
        sentence_select = torch.argmax(sentence_select, dim=-1)

        # Answer Type
        output_answer_type = self.answer_typeout(sequence_output[:, 0, :])
        # Answer Type Loss
        Ltype = None
        if answer_type is not None:
            typeloss = CrossEntropyLoss()
            Ltype = typeloss(output_answer_type, answer_type)
            ans_mask = answer_type > 1
        # Answer Span
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        # Answer Span Loss
        Lspan = None
        if start_positions is not None and end_positions is not None:
            # MRC Loss
            if self.qa_smoothing_method is None or self.qa_smoothing_method == "None":
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                batch_size, ignored_index = start_logits.size()
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
            # Label Smoothing MRC
            else:
                if torch.sum(ans_mask) == 0:
                    start_loss = torch.sum(ans_mask * torch.sum(-torch.log(start_logits.softmax(dim=-1)), dim=-1)) + torch.sum(ans_mask * torch.sum(-torch.log(end_logits.softmax(dim=-1)), dim=-1))
                    end_loss = torch.sum(ans_mask * torch.sum(-torch.log(start_logits.softmax(dim=-1)), dim=-1)) + torch.sum(ans_mask * torch.sum(-torch.log(end_logits.softmax(dim=-1)), dim=-1))
                elif self.qa_smoothing_method == "LabelSmoothing":
                    start_loss = torch.sum(ans_mask * torch.sum(-torch.log(start_logits.softmax(dim=-1)) * label_smoothing_start_label, dim=-1)) / torch.sum(ans_mask)
                    end_loss = torch.sum(ans_mask * torch.sum(-torch.log(end_logits.softmax(dim=-1)) * label_smoothing_end_label, dim=-1)) / torch.sum(ans_mask)
                elif self.qa_smoothing_method == "WordOverlapping":
                    start_loss = torch.sum(ans_mask * torch.sum(-torch.log(start_logits.softmax(dim=-1)) * word_overlapping_start_label, dim=-1)) / torch.sum(ans_mask)
                    end_loss = torch.sum(ans_mask * torch.sum(-torch.log(end_logits.softmax(dim=-1)) * word_overlapping_end_label, dim=-1)) / torch.sum(ans_mask)
                elif self.qa_smoothing_method == "F1Smoothing":
                    start_loss = torch.sum(ans_mask * torch.sum(-torch.log(start_logits.softmax(dim=-1)) * F1_smoothing_start_label, dim=-1)) / torch.sum(ans_mask)
                    end_loss = torch.sum(ans_mask * torch.sum(-torch.log(end_logits.softmax(dim=-1)) * F1_smoothing_end_label, dim=-1)) / torch.sum(ans_mask)
            Lspan = start_loss + end_loss
        # Total Loss
        total_loss = None
        if Lsentence is not None and Ltype is not None and Lspan is not None:
            total_loss = Lsentence + Lspan + Ltype
        return {
            "loss": total_loss,
            "type_logits": output_answer_type,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "sentence_predictions": sentence_select,
        }
