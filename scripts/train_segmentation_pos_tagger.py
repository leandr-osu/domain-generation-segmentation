#!/usr/bin/env python3
"""Train the Universal Dependencies POS checkpoint used for segmentation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np


def read_conllu(path: Path) -> list[list[tuple[str, str]]]:
    from conllu import parse_incr

    sentences: list[list[tuple[str, str]]] = []
    with path.open(encoding="utf-8") as handle:
        for token_list in parse_incr(handle):
            sentence = [
                (str(token["form"]), str(token["upostag"]))
                for token in token_list
                if token.get("form") is not None and token.get("upostag") is not None
            ]
            if sentence:
                sentences.append(sentence)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ud_english_ewt", type=Path, help="UD_English-EWT checkout directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-model", default="bert-base-cased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    import torch
    from torch.utils.data import Dataset, random_split
    from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    sentences = read_conllu(args.ud_english_ewt / "en_ewt-ud-train.conllu")
    sentences += read_conllu(args.ud_english_ewt / "en_ewt-ud-dev.conllu")
    tags = sorted({tag for sentence in sentences for _, tag in sentence})
    tag2id = {tag: index for index, tag in enumerate(tags)}
    id2tag = {index: tag for tag, index in tag2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    words = [[word for word, _ in sentence] for sentence in sentences]
    labels = [[tag2id[tag] for _, tag in sentence] for sentence in sentences]
    encodings = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    aligned = []
    for row, row_labels in enumerate(labels):
        previous = None
        values = []
        for word_id in encodings.word_ids(batch_index=row):
            if word_id is None or word_id == previous:
                values.append(-100)
            else:
                values.append(row_labels[word_id])
            previous = word_id
        aligned.append(torch.tensor(values))
    encodings["labels"] = torch.stack(aligned)

    class PosDataset(Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, index):
            return {key: value[index] for key, value in encodings.items()}

    dataset = PosDataset()
    train_size = int(0.8 * len(dataset))
    generator = torch.Generator().manual_seed(args.seed)
    train, validation = random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(tags),
        id2label=id2tag,
        label2id=tag2id,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output.parent / "checkpoints"),
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train, eval_dataset=validation)
    trainer.train()
    trainer.evaluate()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))


if __name__ == "__main__":
    main()
