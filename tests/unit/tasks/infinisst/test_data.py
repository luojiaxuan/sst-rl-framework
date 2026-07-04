import torch

from nemo_rl.tasks.infinisst.data import build_infinisst_datum


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, message_log, **kwargs):
        return torch.tensor([[10, 0, 20, 21]])


def test_build_infinisst_datum_creates_message_and_metadata():
    datum = build_infinisst_datum(
        tokenizer=FakeTokenizer(),
        data={
            "audio_npy_path": "/data/audio.npy",
            "audio_npy_row": 7,
            "chunk_frame_size": 2,
            "src_segments": ["hello"],
            "tgt_segments": ["你好"],
            "segment_info": [{"start": 0.0, "end": 1.0}],
        },
        src_lang="en",
        tgt_lang="zh",
        task_name="infinisst",
        idx=3,
        multiplier=4,
    )

    assert datum["idx"] == 3
    assert datum["task_name"] == "infinisst"
    assert datum["length"] == 4
    assert datum["message_log"][0]["features"] == ("/data/audio.npy", 7)
    assert datum["message_log"][1]["content"] == "<|video_pad|>" * 8
    assert datum["extra_env_info"]["step"] == 0
    assert datum["extra_env_info"]["chunk_frame_size"] == 8
