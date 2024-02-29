from transformers import ElectraPreTrainedModel, ElectraModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import ipdb


########## ElectraRetriever ##########
class ElectraRetriever(ElectraPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.smoothing_weight = 0
        self.epoch = 0
        self.electra = ElectraModel(config)
        # Self-define layer
        self.single_document_classifier_layer = nn.Linear(config.hidden_size, 2)
        self.double_document_classifier_layer = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
        self,
        question_ids=None,
        document_ids=None,
        question_length=None,
        doc_length=None,
        gold_doc_pair=None,
        gold_answer_doc=None,
        doc_num=None,
    ):
        r"""
        question_ids: (Batch Size, Question Length)
        document_ids: (Batch Size, Document Number, Document Length)
        question_length: (Batch Size, 1)
        document_length: (Batch Size, Document Number)
        gold_doc_pair: index of two related document (Batch Size, 2)
        gold_answer_doc: index of the document which contains answer (Batch Size, 1)
        doc_num: document number in each batch (Batch Size)
        """
        batch_size = len(document_ids)
        device = document_ids.device
        total_rank_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_document_pair_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_retriever_loss = torch.tensor(0.0, device=device, requires_grad=True)
        seleced_document_pair_list = torch.zeros((batch_size, 2), device=device, dtype=torch.long)
        for b in range(batch_size):
            # 2 documents do not need to be calculated
            if doc_num[b] < 3:
                seleced_document_pair_list[b] = torch.tensor((0, 1), device=device, dtype=torch.long)
                continue

            question_document_ids_length = question_length[b].repeat(doc_num[b]) + doc_length[b][: doc_num[b]]
            ids_length = question_document_ids_length.clamp(0, 512).max()
            question_document_ids = torch.zeros((doc_num[b], ids_length.item()), device=device, dtype=torch.long)
            question_document_attention_mask = torch.zeros((doc_num[b], ids_length.item()), device=device, dtype=torch.long)
            for i in range(doc_num[b]):
                question_document_ids[i, : question_document_ids_length[i]] = torch.cat((question_ids[b][: question_length[b]], document_ids[b][i][: doc_length[b][i]]))[:512]  # (Question Length + Document Length)
                question_document_attention_mask[i, : question_document_ids_length[i]] = 1
            stage_one_outputs = self.electra(input_ids=question_document_ids, attention_mask=question_document_attention_mask, return_dict=True)

            document_probability = self.single_document_classifier_layer(torch.index_select(stage_one_outputs[0], dim=1, index=torch.tensor([0], device=device)))
            _, seleced_documents = torch.topk(document_probability.softmax(dim=-1).squeeze()[:, 1], 3)  # Select 3 documents

            # Independent Rank Loss
            if gold_doc_pair is not None:
                document_labels = torch.zeros((doc_num[b]), device=device, dtype=torch.long)
                for gold_doc in gold_doc_pair[b]:
                    document_labels[gold_doc] = 1
                answer_distribution_labels = torch.zeros((doc_num[b]), device=device, dtype=torch.long)
                if gold_answer_doc[b] != -1:
                    answer_distribution_labels[gold_answer_doc[b]] = 1
                label_smoothing_labels = torch.ones((doc_num[b]), device=device, dtype=torch.float) / doc_num[b]
                smoothing_labels = (1 - self.smoothing_weight) * document_labels + self.smoothing_weight * label_smoothing_labels
                rank_loss = (torch.sum(-torch.log(document_probability.softmax(dim=-1).squeeze()[:, 1]) * smoothing_labels.squeeze(), dim=-1) + torch.sum(-torch.log(document_probability.softmax(dim=-1).squeeze()[:, 0]) * (1 - document_labels.squeeze()), dim=-1)) / doc_num[b]
                total_rank_loss = total_rank_loss + torch.mean(rank_loss)

            # Select document pair
            question_double_document_ids_length = torch.tensor(
                [
                    question_length[b] + doc_length[b][seleced_documents[0]] + doc_length[b][seleced_documents[1]],
                    question_length[b] + doc_length[b][seleced_documents[0]] + doc_length[b][seleced_documents[2]],
                    question_length[b] + doc_length[b][seleced_documents[1]] + doc_length[b][seleced_documents[2]],
                ],
                device=device,
                dtype=torch.long,
            )
            ids_length = question_double_document_ids_length.clamp(0, 512).max()
            question_double_document_ids = torch.zeros((3, ids_length.item()), device=device, dtype=torch.long)
            question_double_document_attention_mask = torch.zeros((3, ids_length.item()), device=device, dtype=torch.long)
            doc_pair = torch.zeros((3, 2), device=device, dtype=torch.long)

            for ii in range(3):
                for jj in range(ii + 1, 3):
                    question_double_document_ids[ii + jj - 1][: question_double_document_ids_length[ii + jj - 1]] = torch.cat(
                        (
                            question_ids[b][: question_length[b]],
                            document_ids[b][seleced_documents[ii].item()][: doc_length[b][seleced_documents[ii].item()]],
                            document_ids[b][seleced_documents[jj].item()][: doc_length[b][seleced_documents[jj].item()]],
                        ),
                        dim=-1,
                    )[:512]
                    # (Question Length + Document1 Length + Document2 Length)
                    question_double_document_attention_mask[ii + jj - 1][: question_double_document_ids_length[ii + jj - 1]] = 1
                    doc_pair[ii + jj - 1] = torch.tensor([seleced_documents[ii].item(), seleced_documents[jj].item()], device=device, dtype=torch.long)
            # Double Document Selection Model
            stage_two_outputs = self.electra(input_ids=question_double_document_ids, attention_mask=question_double_document_attention_mask, return_dict=True)
            double_document_probability = self.double_document_classifier_layer(torch.index_select(stage_two_outputs[0], dim=1, index=torch.tensor([0], device=device)))
            # Select Document Pair
            seleced_document_pair = torch.argmax(double_document_probability.squeeze().softmax(dim=-1))
            seleced_document_pair_list[b] = doc_pair[seleced_document_pair.item()]

            # Document Pair Loss
            if gold_doc_pair is not None:
                double_document_label = torch.zeros((3), device=device, dtype=torch.long)
                for ii in range(3):
                    for jj in range(ii + 1, 3):
                        if set([seleced_documents[ii].item(), seleced_documents[jj].item()]) == set(gold_doc_pair[b].tolist()):
                            double_document_label[ii + jj - 1] = 1
                document_pair_loss = torch.mean(torch.sum(-torch.log(double_document_probability.squeeze().softmax(dim=-1)) * double_document_label, dim=-1))
                total_document_pair_loss = total_document_pair_loss + document_pair_loss

        # Retriever Loss
        total_retriever_loss = (total_rank_loss + total_document_pair_loss) / batch_size

        # print({"rank loss": total_rank_loss, "document pair loss": total_document_pair_loss, "retriever loss": total_retriever_loss, "seleced document pair": seleced_document_pair_list, "gold_doc_pair": gold_doc_pair})
        return {"loss": total_retriever_loss, "selected_pair": seleced_document_pair_list}
