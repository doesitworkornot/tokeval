"""Module contains the Embedder classes for embedding datasets.

Specifically for Named Entity Recognition (NER) and Relation Extraction (RE) tasks.
The NEREmbedder class processes datasets for NER tasks,
while the REEmbedder class processes datasets for RE tasks.
Both classes utilize a pre-trained model and tokenizer to generate
embeddings for the respective tasks, and provide methods to retrieve
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class Embedder:
    """Base Embedder class for embedding datasets for token classification tasks."""

    def __init__(
        self,
        dataset_path: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cutoff: int | None = None,
    ) -> None:
        """Initialize the Embedder class for embedding datasets for token classification tasks.

        Args:
            dataset_path: The path to the dataset to be embedded.
            model: The pre-trained model to be used for embedding.
            tokenizer: The tokenizer corresponding to the pre-trained model.
            cutoff: An optional integer to limit the number of samples processed from the dataset.

        """
        self.dataset_path = Path(dataset_path)

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.cutoff = cutoff

        self.label2id = self._load_json("labels.json")
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_set = self._load_jsonl("train/train.jsonl")
        self.val_set = self._load_jsonl("val/val.jsonl")

    def _load_json(self, name: str) -> dict:
        with open(self.dataset_path / name, encoding="utf-8") as f:
            return json.load(f)

    def _load_jsonl(self, rel_path: str) -> Dataset:
        path = self.dataset_path / rel_path

        data = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.cutoff and i >= self.cutoff:
                    break
                data.append(json.loads(line))

        return Dataset.from_list(data)

    def _save_parquet(
        self,
        generator: Iterator[dict],
        save_path: str,
        schema: pa.Schema,
    ) -> int:
        writer = pq.ParquetWriter(save_path, schema)
        count = 0

        try:
            batch = []
            batch_size = 4096

            for row in generator:
                batch.append(row)
                count += 1

                if len(batch) >= batch_size:
                    table = pa.Table.from_pylist(batch, schema=schema)
                    writer.write_table(table)
                    batch.clear()

            if batch:
                table = pa.Table.from_pylist(batch, schema=schema)
                writer.write_table(table)

        finally:
            writer.close()

        return count

    def _load_parquet(self, path: str) -> Dataset:
        return load_dataset(
            "parquet",
            data_files=path,
            split="train",
            streaming=True,
        ).with_format("torch")


class NEREmbedder(Embedder):
    """Embedder class for Named Entity Recognition (NER) tasks."""

    def __init__(
        self,
        dataset_path: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cutoff: int | None = None,
    ) -> None:
        """Initialize the NEREmbedder class for embedding datasets for NER tasks.

        Args:
            dataset_path: The path to the dataset to be embedded.
            model: The pre-trained model to be used for embedding.
            tokenizer: The tokenizer corresponding to the pre-trained model.
            cutoff: An optional integer to limit the number of samples processed from the dataset.

        """
        super().__init__(dataset_path, model, tokenizer, cutoff)

        self.vectorized_train = self._vectorize(self.train_set, "vectorized_train.parquet")
        self.vectorized_val = self._vectorize(self.val_set, "vectorized_val.parquet")

    def get_embeddings(self) -> tuple[Dataset, Dataset]:
        """Return embedded datasets."""
        return self.vectorized_train, self.vectorized_val

    def _tokenize(self, batch: dict) -> dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            batch["tokens"],
            truncation=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )

        labels = []
        for i, word_labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(i)
            aligned = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)
                elif wid != prev:
                    aligned.append(word_labels[wid])
                else:
                    aligned.append(-100)
                prev = wid
            labels.append(aligned)
        tokenized["labels"] = labels
        return tokenized

    def _vectorize(self, dataset: Dataset, save_path: str) -> Dataset:
        schema = pa.schema(
            [
                ("embedding", pa.list_(pa.float32())),
                ("labels", pa.int64()),
            ],
        )

        def generator() -> Iterator[dict]:
            batch_size = 128
            hidden_size = None

            for start in tqdm(range(0, len(dataset), batch_size), desc="NER embedding"):
                batch = dataset[start : start + batch_size]
                tokens = batch["tokens"]
                ner_tags = batch["ner_tags"]

                tok = self.tokenizer(
                    tokens,
                    truncation=True,
                    padding=True,
                    is_split_into_words=True,
                    return_tensors="pt",
                )
                aligned_labels = []
                for i, word_labels in enumerate(ner_tags):
                    word_ids = tok.word_ids(batch_index=i)
                    prev = None
                    aligned = []

                    for wid in word_ids:
                        if wid is None:
                            aligned.append(-100)
                        elif wid != prev:
                            aligned.append(word_labels[wid])
                        else:
                            aligned.append(-100)

                        prev = wid

                    aligned_labels.append(aligned)
                input_ids = tok["input_ids"].to(self.device)
                attention_mask = tok["attention_mask"].to(self.device)

                with torch.no_grad():
                    hidden = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).last_hidden_state.cpu()

                if hidden_size is None:
                    hidden_size = hidden.shape[-1]
                    self.hidden_size = hidden_size
                ignore_label = -100
                for i in range(hidden.shape[0]):
                    for j in range(hidden.shape[1]):
                        label = aligned_labels[i][j]

                        if label == ignore_label:
                            continue

                        yield {
                            "embedding": hidden[i, j].tolist(),
                            "labels": int(label),
                        }

        count = self._save_parquet(generator(), save_path, schema)

        self.ds_len = count

        return self._load_parquet(save_path)


class REEmbedder(Embedder):
    """Embedder class for Relation Extraction (RE) tasks."""

    def __init__(
        self,
        dataset_path: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cutoff: int | None = None,
    ) -> None:
        """Initialize the REEmbedder class for embedding datasets for RE tasks.

        Args:
            dataset_path: The path to the dataset to be embedded.
            model: The pre-trained model to be used for embedding.
            tokenizer: The tokenizer corresponding to the pre-trained model.
            cutoff: An optional integer to limit the number of samples processed from the dataset.

        """
        super().__init__(dataset_path, model, tokenizer, cutoff)

        self.vectorized_train = self._vectorize(self.train_set, "vectorized_train.parquet")
        self.vectorized_val = self._vectorize(self.val_set, "vectorized_val.parquet")

    def get_embeddings(self) -> tuple[Dataset, Dataset]:
        """Return embedded datasets."""
        return self.vectorized_train, self.vectorized_val

    def _extract(self, sentence: str) -> tuple[str, str, str]:
        e1s = sentence.index("<e1>")
        e1e = sentence.index("</e1>")

        e2s = sentence.index("<e2>")
        e2e = sentence.index("</e2>")

        e1 = sentence[e1s + 4 : e1e]
        e2 = sentence[e2s + 4 : e2e]

        clean = sentence.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")

        return clean, e1, e2

    def _vectorize(self, dataset: Dataset, save_path: str) -> Dataset:
        schema = pa.schema(
            [
                ("e1_embedding", pa.list_(pa.float32())),
                ("e2_embedding", pa.list_(pa.float32())),
                ("label", pa.int64()),
            ],
        )

        def generator() -> Iterator[dict]:
            for item in tqdm(dataset, desc="RE embedding"):
                clean, e1, e2 = self._extract(item["sentence"])

                enc = self.tokenizer(
                    clean,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    truncation=True,
                )

                offsets = enc["offset_mapping"][0].tolist()

                enc = {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}

                with torch.no_grad():
                    hidden = self.model(**enc).last_hidden_state[0].cpu()

                def __find(entity: str, clean: str, offsets: list[tuple[int, int]]) -> int | None:
                    pos = clean.index(entity)

                    for i, (start, end) in enumerate(offsets):
                        if start <= pos < end:
                            return i

                    return None

                i1 = __find(e1, clean, offsets)
                i2 = __find(e2, clean, offsets)

                if i1 is None or i2 is None:
                    continue

                yield {
                    "e1_embedding": hidden[i1].tolist(),
                    "e2_embedding": hidden[i2].tolist(),
                    "label": item["relation"],
                }

        count = self._save_parquet(generator(), save_path, schema)

        self.ds_len = count
        self.hidden_size = len(next(generator())["e1_embedding"])

        return self._load_parquet(save_path)
