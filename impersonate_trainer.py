import logging
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

timestamp = datetime.now().strftime("%Y%m%dT%H%M")
logging.basicConfig(
    filename=f"train_{timestamp}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
sanity_check_seeds = [
    "I woke up early this morning in order to",
    "After a particularly warm day, the sun finally began to set. I felt relieved because",
    "When I had attained the age of seventeen my parents resolved that I should become a student at the university of Ingolstadt. I had hitherto attended the schools of Geneva, but ",  # From Frankenstein
    "No man prefers to sleep two in a bed. In fact, you would a good deal rather not sleep with your own brother. I don’t know how it is, but people like to be private when they are sleeping. And when",  # From Moby Dick
]


class ImpersonateTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        name: str,
        loss_ignore_token: int,
        print_every: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.name = name
        self.loss_ignore_token = loss_ignore_token
        self.print_every = print_every
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.loss_log_train = list()
        self.loss_log_eval = list()
        self.birth = time.time()

    def train_one_epoch(self) -> None:
        """Execute one full training epoch"""
        self.model.train()
        loss_log = list()
        for i, (data, target) in enumerate(self.train_loader):
            if i > 10:  # TODO: delete
                break
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(data).logits
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                target.view(B * T),
                ignore_index=self.loss_ignore_token,
            )
            loss_log.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # Intentionally update per batch rather than per epoch
            self.print_progress(i, loss_log, True, 5)
        self.loss_log_train.append(np.array(loss_log).mean())

    def eval_one_epoch(self) -> None:
        self.model.eval()
        loss_log = list()
        for i, (data, target) in enumerate(self.eval_loader):
            if i > 10:  # TODO: delete
                break
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                logits = self.model(data).logits
                B, T, C = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, C),
                    target.view(B * T),
                    ignore_index=self.loss_ignore_token,
                )
            loss_log.append(loss.item())
            self.print_progress(i, loss_log, False, 5)
        self.loss_log_eval.append(np.array(loss_log).mean())

    def stopwatch(self) -> float:
        """Return time elapsed since inception in minutes"""
        time_elapsed = time.time() - self.birth
        return time_elapsed / 60

    def print_progress(
        self,
        iteration: int,
        loss_log: list[float],
        is_train: bool,
        ma_size: int,
    ) -> None:
        if iteration % self.print_every == 0:
            mode = "Train" if is_train else "Eval"
            moving_avg = np.array(loss_log[-ma_size:]).mean()
            logger.info(
                f"Batch {iteration:4} | "
                f"Stopwatch = {self.stopwatch():5.1f} min | "
                f"{mode:5} loss = {moving_avg:5.2f}"
            )

    def save(self) -> None:
        """Save state dicts of model, optimizer, and scheduler"""
        checkpoint = {
            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, f"{self.name}_{timestamp}.pt")

    def launch(
        self,
        num_epochs: int,
    ) -> None:
        best_loss = np.inf
        for i in range(num_epochs):
            self.train_one_epoch()
            self.eval_one_epoch()
            loss_train = self.loss_log_train[-1]
            loss_eval = self.loss_log_eval[-1]
            logger.info(
                f"Epoch {i:2} | "
                f"Train loss = {loss_train:.3f} | "
                f"Eval  loss = {loss_eval:.3f}"
                f"\n{'-'*80}"
            )
            if loss_eval < best_loss:
                self.save()
                best_loss = loss_eval
