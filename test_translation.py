import logging
import sys

logging.basicConfig(level=logging.DEBUG)

from src.helpers.async_wrappers import _translate_sync

class DummyTranslator:
    pass

class DummyModel:
    def generate(self, **kwargs):
        # return dummy token IDs
        import torch
        return torch.tensor([[10, 20, 30]])
    def parameters(self):
        import torch
        yield torch.nn.Parameter(torch.zeros(1))

class DummyProcessor:
    def __call__(self, text, src_lang, return_tensors):
        import torch
        self.last_src_lang = src_lang
        return {"input_ids": torch.tensor([[1, 2, 3]])}
    def decode(self, tokens, skip_special_tokens):
        return [f"translated text from {getattr(self, 'last_src_lang', 'unknown')}"]

class DummyPipeline:
    def __init__(self):
        self.model = DummyModel()
        self.processor = DummyProcessor()
        self.tokenizer = None  # To make it pick up processor

def test():
    pipe = DummyPipeline()
    res = _translate_sync(
        translator=pipe,
        text="روایت سفیر ایران از سخنان پوتین درباره مردم ایران",
        source_lang="fa",
        target_lang="eng",
        model_name="seamless-m4t"
    )
    print(res)

if __name__ == "__main__":
    test()