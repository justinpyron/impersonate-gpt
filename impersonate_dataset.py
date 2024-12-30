from torch.utils.data import Dataset


def process_chunk(chunk: str) -> str:
    first_space = chunk.find(" ")
    last_space = chunk.rfind(" ")
    return chunk[first_space+1: last_space]


def make_chunks(
    text: str,
    length: int,
    skip_lines: int = 400,
) -> list[str]:
    text = "".join(text.splitlines(keepends=True)[skip_lines:])
    chunks = list()
    i = 0
    while i < len(text) - int(1.5*length):
        chunks.append(text[i : i+length])
        chunks.append(text[int(i + 0.5*length) : int(i + 1.5*length)])
        i += length
    chunks = [process_chunk(c) for c in chunks]
    return chunks


class ImposterDataset(Dataset):

    def __init__(
        self,
        file_paths: str,
        characters_per_chunk: int,
        tokenizer,
    ) -> None:
        self.chunks = list()
        for file in file_paths:
            with open(file, "r") as f:
                text = f.read()
            chunk = make_chunks(text, characters_per_chunk)
            self.chunks += [
                tokenizer(c, return_tensors="pt")["input_ids"][0] for c in chunk
            ]
        
    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]
