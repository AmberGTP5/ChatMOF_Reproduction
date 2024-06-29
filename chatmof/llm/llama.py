import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from langchain.base_language import BaseLanguageModel


LLAMA_HF = [
    'Llama-2-7b'
    'Llama-2-7b-hf'
    'Llama-2-7b-chat',
    'Llama-2-7b-chat-hf',
    'Llama-2-13b',
    'Llama-2-13b-hf',
    'Llama-2-13b-chat',
    'Llama-2-13b-chat-hf',
    'Llama-2-70b',
    'Llama-2-70b-hf',
    'Llama-2-70b-chat',
    'Llama-2-70b-chat-hf',
]


def get_llama_llm(model_name: str, max_token: int =4096, temperature: float=0.1) -> BaseLanguageModel:
    #将传入的 model_name 字符串的首字母大写，以确保与 LLAMA 模型的命名规范一致。
    model_name = model_name.capitalize()

    #如果 model_name 不在 LLAMA_HF 列表中，则抛出 ValueError 异常，提示模型名称应该是 LLAMA_HF 列表中的一个。
    if model_name not in LLAMA_HF:
        raise ValueError(f'model_name should be one of {LLAMA_HF}, not {model_name}')
    
    #构建完整的模型名称，格式为 'meta-llama/{model_name}'。
    model_name = f'meta-llama/{model_name}'

    #使用 Hugging Face 的 AutoTokenizer 类从预训练模型中加载分词器。
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            use_auth_token=True,
                                            )
    
    #使用 Hugging Face 的 AutoModelForCausalLM 类从预训练模型中加载语言模型。
    '''
        AutoModelForCausalLM.from_pretrained() 函数通过这些参数加载预训练的语言模型，并根据参数的配置将其加载到合适的设备上，使用适当的数据类型进行计算，并尝试使用身份验证令牌来访问模型。
    '''
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map='auto',
                                                torch_dtype=torch.float16,
                                                use_auth_token=True,
                                                )

    #使用 Hugging Face 的 pipeline 函数创建一个文本生成管道，用于生成文本。
    '''
        pipeline 函数使用这些参数创建了一个文本生成管道，该管道可以使用指定的语言模型和分词器来生成文本，同时还可以配置采样方式、保留的最高概率的 token 数量以及生成的最大 token 数量等参数
    '''
    pipe = pipeline("text-generation",# 要创建的管道类型。在这种情况下，它指定了一个文本生成管道，用于生成文本。
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                do_sample=True,
                top_k=30,
                max_new_tokens=max_token,
                eos_token_id=tokenizer.eos_token_id
                )

    #使用 Hugging Face 的 HuggingFacePipeline 类创建一个语言模型实例，传入了生成文本的管道和额外的模型参数（例如温度）。
    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':temperature})

    #返回创建的语言模型实例（llm）
    return llm



