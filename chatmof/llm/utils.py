from chatmof.llm.llama import get_llama_llm
from chatmof.llm.openai import get_openai_llm
from chatmof.llm.codellama import get_codellama_llm
from langchain.base_language import BaseLanguageModel


'''
    get_llm 函数根据给定的模型名称，选择并返回相应的语言模型实例。它根据模型名称的前缀来判断模型类型，然后调用相应的函数获取语言模型实例。
    函数参数 temperature 和 max_tokens 用于控制生成文本的多样性和长度。函数确保了输入的模型名称是合法的，如果不是，会抛出 ValueError 异常。
'''
#-> BaseLanguageModel 表示 get_llm 函数应该返回一个 BaseLanguageModel 类型的对象，这有助于理解函数的预期行为并提供代码编辑器的支持。
def get_llm(model_name: str, temperature: float = 0.1, max_tokens: int = 4096) -> BaseLanguageModel:
    m_name = model_name.lower()

    #如果 model_name 以 'gpt' 开头，则调用 get_openai_llm 函数获取 OpenAI 的语言模型实例。
    if m_name.startswith('gpt'):
        llm = get_openai_llm(model_name, temperature=temperature)

    #如果 model_name 以 'llama' 开头，则调用 get_llama_llm 函数获取 Llama 的语言模型实例
    elif m_name.startswith('llama'):
        llm = get_llama_llm(model_name, max_token=max_tokens, temperature=temperature)

    #如果 model_name 以 'codellama' 开头，则调用 get_codellama_llm 函数获取 CodeLlama 的语言模型实例。
    elif m_name.startswith('codellama'):
        llm = get_codellama_llm(model_name, max_token=max_tokens, temperature=temperature)

    #如果 model_name 不是 'gpt'、'llama' 或 'codellama' 中的任何一个，则抛出 ValueError 异常。
    else:
        raise ValueError(f'model_name should be one of llama and gpt, not {model_name}')
    
    return llm
