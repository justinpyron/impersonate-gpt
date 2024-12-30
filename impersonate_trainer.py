import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        save_path: str,
        loss_ignore_token: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.save_path = save_path
        self.loss_ignore_token = loss_ignore_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train_one_epoch(self) -> None:
        """Execute one full training epoch"""
        self.model.train()
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(data).logits
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                target.view(B * T),
                ignore_index=self.loss_ignore_token,
            )
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # Intentionally update per batch rather than per epoch

            # print(f"loss = {loss.item()}")
            # print(f"self.scheduler.get_last_lr() = {self.scheduler.get_last_lr()}")
            # print(80 * "-")

    def eval_one_epoch(self):
        self.model.eval()
        for data, target in self.eval_loader:
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                logits = self.model(data).logits
                B, T, C = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, C),
                    target.view(B * T),
                    ignore_index=self.loss_ignore_token,
                )
                print(f"loss = {loss.item()}")

    def save(self, path: str) -> None:
        """Save state dicts of model, optimizer, and scheduler"""
        checkpoint = {
            "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, path)

    def launch(
        num_steps: int,
    ) -> None:
        pass


# TODO: add MLflow
