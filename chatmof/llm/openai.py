from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.base_language import BaseLanguageModel


CHAT_MODEL = [
    'gpt-4',
    'gpt-4-0613',
    'gpt-4-32k',
    'gpt-4-32k-0613',
    'gpt-4-0314',
    'gpt-4-32k-0314',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-instruct',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-16k-0613',
    'gpt-3.5-turbo-0301',
]


MODEL = [
    'text-davinci-003',
    'text-davinci-002',
]

#该函数根据给定的 model_name 和 temperature 返回一个特定的 OpenAI 语言模型实例。
def get_openai_llm(model_name: str, temperature: float=0.1) -> BaseLanguageModel:
    #将传入的 model_name 字符串转换为小写字母，以确保匹配模型名称时不受大小写影响。
    model_name = model_name.lower()
    
    #如果 model_name 在 CHAT_MODEL 列表中，则使用给定的 model_name 和 temperature 创建一个 ChatOpenAI 实例，并将其赋值给 llm 变量。
    if model_name in CHAT_MODEL:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    #如果 model_name 不在 CHAT_MODEL 中，但在 MODEL 列表中，则使用给定的 model_name 和 temperature 创建一个通用的 OpenAI 实例，并将其赋值给 llm 变量。
    elif model_name in MODEL:
        llm = OpenAI(model_name=model_name, temperature=temperature)

    return llm
