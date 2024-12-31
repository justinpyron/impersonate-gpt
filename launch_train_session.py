from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from impersonate_dataset import ImpersonateDataset
from impersonate_trainer import ImpersonateTrainer
from train_configs import configs

# Model params
MODEL_CARD = "openai-community/gpt2"
# Data params
CHARACTERS_PER_CHUNK = 2000  # GPT2 max sequence length = 1024 tokens
BATCH_SIZE = 8
# Training params
MAX_LR = 1e-5
WARMUP_STEPS = 200
WARM_UP_START_FACTOR = 1e-2
GAMMA = 0.9995
PRINT_EVERY = 20
GENERATE_EVERY = 100
SHORT_CIRCUIT = 99999
NUM_EPOCHS = 10


def collate(example, pad_token_id):
    data = pad_sequence(
        (x for (x, y) in example), padding_value=pad_token_id, batch_first=True
    )
    target = pad_sequence(
        (y for (x, y) in example), padding_value=pad_token_id, batch_first=True
    )
    return data, target


def get_dataloaders(
    files_train: list[str],
    files_eval: list[str],
    pad_token_id: int,
    tokenizer,
) -> tuple[DataLoader, DataLoader]:
    dataset_train = ImpersonateDataset(
        files_train,
        characters_per_chunk=CHARACTERS_PER_CHUNK,
        tokenizer=tokenizer,
    )
    dataset_eval = ImpersonateDataset(
        files_eval,
        characters_per_chunk=CHARACTERS_PER_CHUNK,
        tokenizer=tokenizer,
    )
    loader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: collate(x, pad_token_id),
    )
    loader_eval = DataLoader(
        dataset_eval,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: collate(x, pad_token_id),
    )
    return loader_train, loader_eval


def get_trainer(config):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD)
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    loader_train, loader_eval = get_dataloaders(
        config.files_train, config.files_eval, pad_token_id, tokenizer
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_CARD)
    optimizer = AdamW(model.parameters(), lr=MAX_LR)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(
                optimizer,
                start_factor=WARM_UP_START_FACTOR * MAX_LR,
                end_factor=1,
                total_iters=WARMUP_STEPS,
            ),
            ExponentialLR(optimizer, gamma=GAMMA),
        ],
        milestones=[WARMUP_STEPS],
    )
    trainer = ImpersonateTrainer(
        model,
        tokenizer,
        optimizer,
        scheduler,
        loader_train,
        loader_eval,
        name=config.name,
        pad_token_id=pad_token_id,
        print_every=PRINT_EVERY,
        generate_every=GENERATE_EVERY,
        short_circuit=SHORT_CIRCUIT,
    )
    return trainer


if __name__ == "__main__":
    # config = configs["fitzgerald"]
    config = configs["twain"]
    trainer = get_trainer(config)
    trainer.launch(NUM_EPOCHS)

