"""Provide functionnality to apply train the model dataset."""

import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pandas import DataFrame as D
from pandas import Series as S
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from src.models import cross_encoder_dataset
from src.models.cross_encoder_factory import CrossEncoderModelFactory
from src.models.bert_model import BertModel


class CrossEncoderModel:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        tokenizer_path: str,
        num_labels: int,
        max_seq_length: int,
        batch_size: int,
        epochs: int,
        lr: float,
        device: Any,
    ):

        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_seq_length = max_seq_length
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device

        self.cross_encoder_model_factory = CrossEncoderModelFactory()

        # get model and tokenize
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            print(f"Loading {self.model_path} from {self.model_path}")
            model, tokenizer = self.cross_encoder_model_factory.load_model(
                self.model_name, self.model_path, self.tokenizer_path
            )
        else:
            print(f"Instantiating {self.model_path} and saving it to {self.model_path}")
            model, tokenizer = self.cross_encoder_model_factory.create_model(
                self.model_name
            )
            self.cross_encoder_model_factory.save_model(
                model=self.model,
                tokenizer=self.tokenizer,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
            )

        # Create cross encoder model
        self.model = BertModel(model=model, num_labels=num_labels)
        self.tokenizer = tokenizer

        # run the model on device
        self.model.to(device)

        # Get Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

    def fit(self, x_train: D, y_train: S, x_val: D, y_val: S):

        # Create train dataset
        train_dataset = cross_encoder_dataset.CrossEncoderDataset(
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            x=x_train,
            y=y_train,
        )

        # Create val dataset
        val_dataset = cross_encoder_dataset.CrossEncoderDataset(
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            x=x_val,
            y=y_val,
        )

        # Create Data Loader
        train_dataloader = DataLoader(
            dataset=train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.batch_size,
        )

        # Create Data Loader
        val_dataloader = DataLoader(
            dataset=val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=self.batch_size,
        )

        # Total number of training steps
        total_steps = len(train_dataloader) * self.epochs

        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        self.__train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)

    def predict(self, x_test):

        proba = self.predict_proba(x_test)

        y_pred = np.argmax(proba, axis=-1)

        return y_pred

    def save_model(self, model_path: str):

        print(f"Ssave trained model to {model_path}")

        self.cross_encoder_model_factory.save_model(
            model=self.model, model_path=model_path
        )

    def predict_proba(self, x_test):

        # Create val dataset
        test_dataset = cross_encoder_dataset.CrossEncoderDataset(
            tokenizer=self.tokenizer, max_seq_length=self.max_seq_length, x=x_test
        )

        # Create test data Loader
        test_dataloader = DataLoader(
            dataset=test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=self.batch_size,
        )

        prediction = []

        # For each batch in our test set...
        for batch in test_dataloader:

            # Load batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)

            prediction.append(logits)

        # Concatenate logits from each batch
        prediction = torch.cat(prediction, dim=0)

        # Apply softmax to calculate probabilities
        proba = F.softmax(prediction, dim=1).cpu().numpy()

        return proba

    def __train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
    ):

        """Train the croos encode model."""

        # Start training loop
        print("Start training cross encoder\n")

        for epoch_i in range(self.epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}"
            )
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1

                # Load batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Zero out any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = self.model(input_ids, attention_mask)

                # Compute loss and accumulate the loss values
                loss = self.loss_fn(logits, labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                self.optimizer.zero_grad()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (
                    step == len(train_dataloader) - 1
                ):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}"
                    )

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            if val_dataloader is not None:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy = self.__evaluate_on_val(
                    val_dataloader=val_dataloader
                )

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}"
                )
                print("-" * 70)
            print("\n")

        print("Cross encoder training completed!")

    def __evaluate_on_val(self, val_dataloader: DataLoader):

        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        self.model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        # For each batch in our validation set...
        for batch in val_dataloader:

            # Load batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)

            # Compute loss
            loss = self.loss_fn(logits, labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = preds == labels
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)

        val_accuracy = torch.cat(val_accuracy)
        val_accuracy = val_accuracy.detach().cpu().numpy()
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy
