import os
import csv
from typing import Dict, List

from sklearn.metrics import classification_report, mean_absolute_error
from logzero import logger
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from transformers import AdamW
from torch.optim import SGD
from torch.utils.data import DataLoader

from radie.src.utils.trainer_utils import to_tensor, to_device

def get_linear_schedule_with_warmup(optimizer, num_training_steps):

    def lr_lambda(current_step: int):
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps))
        )

    return LambdaLR(optimizer, lr_lambda, -1)


class BaseTrainer():
    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"running on device {self.device}")

        self.model.to(self.device)

    def build_data_loader(self, dataset: List, batch_size: int,
                          shuffle: bool) -> DataLoader:
        """ 入力配列からloaderを作成する """
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=to_tensor)

    def save_model(self, output_dir: str) -> None:
        """ モデルを指定されたディレクトリに保存する """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def create_optimizer_and_scheduler(self,
                                       num_training_steps: int,
                                       lr: float = 5e-5):
        """ オプティマイザーとスケジューラを指定する """

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=lr,
                               betas=(0.9, 0.999))

        # self.scheduler = ReduceLROnPlateau(self.optimizer,
        #                                    factor=0.1,
        #                                    patience=2,
        #                                    mode='min')

        # for param in self.model.bert.parameters():
        #     param.requires_grad = False
        # self.model.classifier.requires_grad = True

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_training_steps=num_training_steps)

    def train(
        self,
        train_loader: DataLoader,
        lr: int,
        num_epochs: int
    ):

        logger.info("***** Running Training *****")

        num_training_steps = len(train_loader) * num_epochs

        self.create_optimizer_and_scheduler(num_training_steps, lr)

        torch.backends.cudnn.benchmark = True

        for epoch in range(num_epochs):

            self.model.train()

            epoch_loss = 0.0
            epoch_acc = 0
            num_examples = 0

            for batch in train_loader:

                inputs = to_device(self.device, batch)

                outputs = self.model(**inputs)

                _, preds = torch.max(outputs.logits, 1)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.zero_grad()
                outputs.loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss += outputs.loss.item()
                epoch_acc += torch.sum(
                    preds == inputs['labels']).item()

                num_examples += inputs['input_ids'].size(0)

            epoch_loss /= len(train_loader)
            epoch_acc /= num_examples

            logger.info(f'Epoch {epoch + 1} | Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        torch.cuda.empty_cache()

    def predict(self, pred_loader: DataLoader, logs_dir=False):

        logger.info("***** Running Prediction *****")

        self.model.eval()

        inputs_list = list()
        preds_list = list()
        probs_list = list()

        with torch.no_grad():
            for batch in pred_loader:

                inputs = to_device(self.device, batch)

                inputs_list.extend(inputs['input_ids'].tolist())

                outputs = self.model(**inputs)
                outputs_probs = F.softmax(outputs.logits, dim=1)

                probs, preds = torch.max(outputs_probs, 1)

                probs_list.extend(probs.tolist())
                preds_list.extend(preds.tolist())

        f = open(os.path.join(logs_dir, 'predictions.csv'), 'w')
        writer = csv.writer(f, lineterminator='\n', delimiter='\t')
        for i in range(len(inputs_list)):
            input_tokens = self.tokenizer.decode(inputs_list[i],
                                                 skip_special_tokens=True)
            writer.writerow([input_tokens, preds_list[i], probs_list[i]])
        f.close()

    def evaluate(self,
                 test_loader: DataLoader,
                 logs_dir,
                 do_error_analysis=False) -> Dict:

        logger.info("***** Running Evalation *****")

        self.model.eval()

        inputs_list = list()
        trues_list = list()
        preds_list = list()

        with torch.no_grad():
            for batch in test_loader:

                inputs = to_device(self.device, batch)

                inputs_list.extend(inputs['input_ids'].tolist())
                trues_list.extend(inputs['labels'].tolist())

                outputs = self.model(**inputs)

                _, preds = torch.max(outputs.logits, 1)

                preds_list.extend(preds.tolist())

        logger.info(
            f'\n{classification_report(trues_list, preds_list, digits=4)}')

        logger.info(
            f'mean absolute error : {mean_absolute_error(trues_list, preds_list):4f}'
        )

        if do_error_analysis:
            f = open(os.path.join(logs_dir, 'error_analysis.csv'), 'w')
            writer = csv.writer(f, lineterminator='\n', delimiter='\t')
            for i in range(len(inputs_list)):
                input_tokens = self.tokenizer.decode(inputs_list[i],
                                                     skip_special_tokens=True)
                writer.writerow([input_tokens, trues_list[i], preds_list[i]])
            f.close()

            logger.info(f'output error analysis result')
            logger.info(os.path.join(logs_dir, 'error_analysis.csv'))

        metrics = classification_report(trues_list,
                                        preds_list,
                                        output_dict=True)

        metrics['mae'] = mean_absolute_error(trues_list, preds_list)

        return metrics
