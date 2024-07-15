# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The EmbeddingsLLM class."""
import transformers.modeling_outputs
from transformers import AutoTokenizer, AutoModel
from typing_extensions import Unpack
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)

from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes


def execute_chatglm3_6b_embeddings_llm(
        input: EmbeddingInput,
        **kwargs
) -> EmbeddingOutput | None:
    """
    创建chatglm3_6b模型
    :param input: 输入文本;
    :param kwargs: 创建模型需要的参数配置;

    :return: 词嵌入;
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

    tensors = {
        key: value.to(_kwargs['device']) for key, value in tokenizer(input, return_tensors='pt', padding=True).items()
    }
    outputs = model(
            **tensors,
            output_hidden_states=True
    )

    embeddings = outputs.hidden_states[0][:, :, -1]

    return [e.detach().cpu().numpy().tolist() for e in embeddings]


# 注册的可执行模型
llm_executors = {
    'chatglm3_6b': execute_chatglm3_6b_embeddings_llm,
}


class CustomEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
            self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        args.update(self.configuration.raw_config)
        executor = llm_executors.get(args['model'])
        return executor(
            input=input,
            **args
        )