# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A text-completion based LLM."""

import logging

from typing_extensions import Unpack
from transformers import AutoTokenizer, AutoModel
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)

from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes
from .utils import get_completion_llm_args

log = logging.getLogger(__name__)


class ChatGLM36BLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A text-completion based LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

        self.tokenizer = AutoTokenizer.from_pretrained(self.configuration.model_path, trust_remote_code=True)
        self.history = []
        self.past_key_values = None

    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        input = input.format(**kwargs.get('variables'))
        res: CompletionOutput = ""
        current_length = 0
        self.model = AutoModel.from_pretrained(self.configuration.model_path, trust_remote_code=True, device='cuda')
        self.model.eval()
        for response, self.history, self.past_key_values in self.model.stream_chat(
            self.tokenizer,
            input,
            history=self.history,
            past_key_values=self.past_key_values,
            return_past_key_values=True
        ):
            res += response[current_length:]
            current_length = len(response)


        print('========================================')
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        print(f'input:\n{input}\n')
        print(f'res:\n{res}\n')
        print('========================================')
        return res