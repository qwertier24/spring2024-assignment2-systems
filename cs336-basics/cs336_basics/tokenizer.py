import os
from typing import Tuple, Mapping, Optional, Iterable, Iterator
import regex as re
import collections
import dataclasses
import queue
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@dataclasses.dataclass
class HeapNode:
    freq: int
    byte_pair: Tuple[bytes, bytes]

    def __lt__(self, other):
        return self.freq > other.freq or (
            self.freq == other.freq and self.byte_pair > other.byte_pair
        )


class BPETokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        """Given the path to a JSON vocab, a file with BPE merges, and a list of special tokens,
        return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

        Args:
            vocab: dict[int, bytes]
                The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
            special_tokens: Optional[list[str]]
                A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.
        """
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = list() if special_tokens is None else special_tokens
        self._special_tokens.sort(key=lambda x: len(x), reverse=True)
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}
        self._merge_to_id = {merge: i for i, merge in enumerate(merges)}

    def split_text_with_special_token(self, texts: list[str], token: str):
        new_texts = []
        for text in texts:
            splited_text = text.split(token)
            for i, t in enumerate(splited_text):
                if i > 0:
                    new_texts.append(token)
                new_texts.append(t)
        return new_texts

    def encode(self, text):
        ids = []
        if self._special_tokens:
            texts = re.split(
                "(" + "|".join([re.escape(t) for t in self._special_tokens]) + ")", text
            )
        else:
            texts = [text]
        for text in texts:
            if not text:
                continue
            if text in self._special_tokens:
                ids.append(self._reverse_vocab[text.encode()])
                continue
            pre_tokens = re.findall(PAT, text)
            for pre_token in pre_tokens:
                if pre_token in self._special_tokens:
                    ids.append(self._reverse_vocab[pre_token.encode()])
                pre_token = pre_token.encode()
                pre_token = [pre_token[i : i + 1] for i in range(len(pre_token))]
                # for j, merge in enumerate(self._merges):
                #     new_pre_token = []
                #     for i in range(0, len(pre_token)):
                #         if (
                #             i + 1 < len(pre_token)
                #             and (pre_token[i], pre_token[i + 1]) == merge
                #         ):
                #             pass
                #         elif i > 0 and (pre_token[i - 1], pre_token[i]) == merge:
                #             new_pre_token.append(pre_token[i - 1] + pre_token[i])
                #         else:
                #             new_pre_token.append(pre_token[i])
                #     pre_token = new_pre_token
                while True:
                    new_pre_token = []
                    unchanged = True
                    smallest_merge_id = None
                    for i in range(0, len(pre_token)):
                        if (
                            i + 1 < len(pre_token)
                            and (pre_token[i], pre_token[i + 1]) in self._merge_to_id
                        ):
                            unchanged = False
                            merge_id = self._merge_to_id[
                                (pre_token[i], pre_token[i + 1])
                            ]
                            smallest_merge_id = (
                                min(smallest_merge_id, merge_id)
                                if smallest_merge_id is not None
                                else merge_id
                            )
                    if unchanged:
                        break
                    for i in range(0, len(pre_token)):
                        if (
                            i + 1 < len(pre_token)
                            and (pre_token[i], pre_token[i + 1])
                            == self._merges[smallest_merge_id]
                        ):
                            pass
                        elif (
                            i > 0
                            and (pre_token[i - 1], pre_token[i])
                            == self._merges[smallest_merge_id]
                        ):
                            unchanged = False
                            new_pre_token.append(pre_token[i - 1] + pre_token[i])
                        else:
                            new_pre_token.append(pre_token[i])
                    pre_token = new_pre_token
                for sub_token in pre_token:
                    # print(pre_token)
                    ids.append(self._reverse_vocab[sub_token])
            # print("ids", ids)
        return ids

    def decode(self, tokens):
        strs = []
        for token in tokens:
            if token in self._vocab:
                strs.append(self._vocab[token])
            else:
                strs.append(b"\xff\xfd")
        return b"".join(strs).decode(errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory."""
        for text in iterable:
            ids = self.encode(text)
            for tid in ids:
                yield tid

    @classmethod
    def _find_tp_max_freq(cls, tp_to_freq):
        max_freq = 0
        max_freq_tp = None
        for tp, freq in tp_to_freq.items():
            if freq > max_freq or (freq == max_freq and freq > 0 and tp > max_freq_tp):
                max_freq = freq
                max_freq_tp = tp
        return max_freq_tp

    @classmethod
    def train_bpe(
        cls, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
    ) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Given the path to an input corpus, run train a BPE tokenizer and
        output its vocabulary and merges.

        Args:
            input_path: str | os.PathLike
                Path to BPE tokenizer training data.
            vocab_size: int
                Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens: list[str]
                A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            Tuple of (vocab, merges):
                vocab: dict[int, bytes]
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges: list[tuple[bytes, bytes]]
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
        """

        pre_token_to_freq = collections.defaultdict(int)
        with open(input_path) as f:
            print("opened file")
            tokens = re.findall(PAT, f.read())
            print("found all tokens", len(tokens))
            for pre_token in tokens:
                pre_token = pre_token.encode("utf-8")
                bs = []
                for i in range(len(pre_token)):
                    bs.append(pre_token[i : i + 1])
                pre_token_to_freq[tuple(bs)] += 1
        print("pre token finished")
        pre_tokens = list()
        pre_tokens_freq = list()
        tp_to_freq: Mapping[Tuple[bytes, bytes], int] = collections.defaultdict(int)
        tp_to_index: Mapping[Tuple[bytes, bytes], set[int]] = collections.defaultdict(
            set
        )
        for pre_token, freq in pre_token_to_freq.items():
            pre_tokens.append(pre_token)
            pre_tokens_freq.append(freq)
            for i in range(1, len(pre_token)):
                tp_to_freq[(pre_token[i - 1], pre_token[i])] += freq
                tp_to_index[(pre_token[i - 1], pre_token[i])].add(len(pre_tokens) - 1)
        del pre_token_to_freq

        vocab = dict()
        for token in special_tokens:
            vocab[len(vocab)] = token.encode()
        for i in range(256):
            vocab[len(vocab)] = i.to_bytes(1)

        merges = list()
        while len(vocab) < vocab_size:
            if len(vocab) % 100 == 0:
                print("%d/%d"%(len(vocab), vocab_size))
            tp_with_max_freq = cls._find_tp_max_freq(tp_to_freq)
            if tp_with_max_freq is None:
                break
            indices = tp_to_index[tp_with_max_freq].copy()
            merges.append(tp_with_max_freq)
            vocab[len(vocab)] = b"".join(tp_with_max_freq)
            # print("before merging", merges[-1], tp_to_freq)
            for i in indices:
                new_pre_token = list()
                for j in range(len(pre_tokens[i])):
                    if (
                        j + 1 < len(pre_tokens[i])
                        and (pre_tokens[i][j], pre_tokens[i][j + 1]) == tp_with_max_freq
                    ):
                        pass
                    elif (
                        j > 0
                        and (pre_tokens[i][j - 1], pre_tokens[i][j]) == tp_with_max_freq
                    ):
                        new_pre_token.append(b"".join(tp_with_max_freq))
                    else:
                        new_pre_token.append(pre_tokens[i][j])
                    if j > 0:
                        tp_to_freq[
                            (pre_tokens[i][j - 1], pre_tokens[i][j])
                        ] -= pre_tokens_freq[i]
                        if i in tp_to_index[(pre_tokens[i][j - 1], pre_tokens[i][j])]:
                            tp_to_index[
                                (pre_tokens[i][j - 1], pre_tokens[i][j])
                            ].remove(i)

                pre_tokens[i] = tuple(new_pre_token)
                for j in range(len(pre_tokens[i])):
                    if j > 0:
                        tp_to_freq[
                            (pre_tokens[i][j - 1], pre_tokens[i][j])
                        ] += pre_tokens_freq[i]
                        tp_to_index[(pre_tokens[i][j - 1], pre_tokens[i][j])].add(i)

        return vocab, merges


if __name__ == "__main__":
    # PATH = "/home/qwertier/projects/stanford_cs336/spring2024-assignment1-basics/tests/fixtures/small_test.txt"
    # print(BPETokenizer.train_bpe(PATH, 1000, [])[0])
    import pathlib

    DATA_PATH = pathlib.Path("/data/stanford_cs336/TinyStoriesV2-GPT4-train.txt")
    VOCAB_PATH = "/data/stanford_cs336/TinyStoriesV2-GPT4-vocab.json"
    MERGES_PATH = "/data/stanford_cs336/TinyStoriesV2-GPT4-merges.txt"
    vocab, merges = BPETokenizer.train_bpe(DATA_PATH, 10000, ["<|endoftext|>"])
    gpt2_encoder = gpt2_bytes_to_unicode()
    with open(MERGES_PATH, "w") as f:
        for merge in merges:
            f.write(' '.join([''.join([gpt2_encoder[b] for b in m]) for m in merge]) + '\n')

    vocab_json = {''.join([gpt2_encoder[b] for b in token]):tid for tid, token in vocab.items()}
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab_json, f)
