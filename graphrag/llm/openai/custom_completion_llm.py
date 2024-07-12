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


def execute_chatglm3_6b_completion_llm(
        input: CompletionInput,
        **kwargs
) -> CompletionOutput | None:
    """
    创建chatglm3_6b模型
    :param input: 输入的prompt;
    :param kwargs: 创建模型需要的参数配置;

    :return: 模型的输出;
    """
    _kwargs = {
        'model_path': ".",
        'trust_remote_code': True,
        'device': 'cuda'
    }
    _kwargs.update(kwargs)

    tokenizer = AutoTokenizer.from_pretrained(_kwargs['model_path'], trust_remote_code=_kwargs['trust_remote_code'])
    model = AutoModel.from_pretrained(
        _kwargs['model_path'],
        trust_remote_code=_kwargs['trust_remote_code'],
        device=_kwargs['device'],
    )
    model.eval()

    output = ''
    history = []
    past_key_values = None
    current_length = 0
    for response, history, past_key_values in model.stream_chat(
            tokenizer,
            input,
            history=history,
            past_key_values=past_key_values,
            return_past_key_values=True
    ):
        output += response[current_length:]
        current_length = len(response)
    return output


# 注册的可执行模型
llm_executors = {
    'chatglm3_6b': execute_chatglm3_6b_completion_llm,
}


class CustomCompletionLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A text-completion based LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
            self,
            input: CompletionInput,
            **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        input = input.format(**kwargs.get('variables'))
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        executor = llm_executors.get(args['model'])
        return executor(
            input=input,
            **args
        )
