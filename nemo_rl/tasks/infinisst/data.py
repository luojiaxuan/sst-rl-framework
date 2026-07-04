# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from typing import Any, Iterator

from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec

CODE2LANG = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}

INSTRUCTION = "Translate the following speech from {} to {}."


def build_infinisst_datum(
    tokenizer: PreTrainedTokenizerBase,
    data: dict[str, Any],
    src_lang: str,
    tgt_lang: str,
    task_name: str,
    idx: int,
    multiplier: int = 1,
) -> DatumSpec:
    import torch

    instruction = INSTRUCTION.format(CODE2LANG[src_lang], CODE2LANG[tgt_lang])
    chunk_frame_size = data["chunk_frame_size"] * multiplier
    message_log = [
        {
            "role": "system",
            "content": instruction,
            "features": (data["audio_npy_path"], data["audio_npy_row"]),
        },
        {
            "role": "user",
            "content": "<|video_pad|>" * chunk_frame_size,
        },
    ]
    token_ids = tokenizer.apply_chat_template(
        message_log,
        return_tensors="pt",
        add_special_tokens=False,
        add_generation_prompt=True,
    )[0]

    system_prompt_end = torch.nonzero(token_ids == tokenizer.eos_token_id)[0]
    message_log[0]["token_ids"] = token_ids[: system_prompt_end + 1]
    message_log[1]["token_ids"] = token_ids[system_prompt_end + 1 :]

    return {
        "message_log": message_log,
        "length": len(token_ids),
        "extra_env_info": {
            "step": 0,
            "chunk_frame_size": chunk_frame_size,
            "src_segments": data["src_segments"],
            "tgt_segments": data["tgt_segments"],
            "segment_info": data["segment_info"],
        },
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
    }


class IterableInfiniSSTDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_file: str,
        shuffle: bool,
        seed: int,
        src_lang: str,
        tgt_lang: str,
        task_name: str,
        length: int,
        multiplier: int,
    ):
        super().__init__()
        import pandas as pd

        self.tokenizer = tokenizer
        self.df = pd.read_parquet(data_file)
        self.shuffle = shuffle
        self.seed = seed
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.task_name = task_name
        self.length = length
        self.multiplier = multiplier

    def __iter__(self) -> Iterator[DatumSpec]:
        print("Starting IterableInfiniSSTDataset.")
        df = self.df.sample(frac=1, random_state=self.seed) if self.shuffle else self.df
        for i in itertools.count():
            yield build_infinisst_datum(
                tokenizer=self.tokenizer,
                data=df.iloc[i % len(df)].to_dict(),
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                task_name=self.task_name,
                idx=i,
                multiplier=self.multiplier,
            )

    def __len__(self):
        return self.length
