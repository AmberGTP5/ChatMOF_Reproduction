import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import argparse
import copy
from itertools import chain
import warnings

from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManagerForChainRun
from chatmof.config import config as config_default
from chatmof.agents.agent import ChatMOF
from chatmof.llm import get_llm

#一个控制警告显示的有用工具，可以在需要时忽略所有警告，从而使输出更简洁、干净。
warnings.filterwarnings(action='ignore')


str_kwargs_names = {
    #accelerator: 描述 MOFTransformer 使用的设备名称，可以是 cuda、gpu 或 cpu。默认值为 cuda。
    'accelerator': "Device name for MOFTransformer. accelerator must be one of [cuda, gpu, cpu] (default: cuda)",
    #logger: 描述用于生成任务的日志记录器，默认值为 generate_mof.log。
    'logger': 'logger for generation task (Default: generate_mof.log)',
}

int_kwargs_names = {
    #max_length_in_predictor: 在预测步骤中 MOF 的最大数量，默认值为 30。
    'max_length_in_predictor': 'max number of MOFs in predictor step. (default: 30)',
    #num_genetic_cycle: 遗传算法循环的次数，默认值为 3。
    'num_genetic_cycle': 'number of genetic algorithm cycle (default: 3)',
}

#目前为空，没有浮点数类型的参数。
float_kwargs_names = {
}

bool_kwargs_names = {
    #如果为 True，则使用 "search internet" 工具，默认值为 False。
    'search_internet': 'If True, using "search internet" tools. (default = False)',
    #如果为 True，则打印中间步骤，默认值为 True。
    'verbose': 'If True, print intermediate step. (default = True)'
}


'''
    该函数从配置文件 config_default 复制配置，并根据传入的关键字参数更新配置。
    它创建并配置了一个语言模型（LLM），并将其传递给 ChatMOF 实例。
    它提示用户输入一个问题，然后将问题传递给 ChatMOF 以获取答案。
    最后，它打印出答案并返回给调用者。
'''
#返回类型标注为 str，表示函数返回一个字符串
def main(model='gpt-4', temperature=0.1, **kwargs) -> str:
    #config 是从 config_default 深度复制过来的，确保对 config 的修改不会影响原始的 config_default。
    config = copy.deepcopy(config_default)

    #使用 kwargs 更新 config 字典。这允许调用者通过关键字参数覆盖默认配置。
    config.update(kwargs)
    #提取 config 中的 search_internet 和 verbose 参数。
    search_internet = config['search_internet']
    verbose = config['verbose']

    #调用 get_llm 函数创建语言模型实例，使用 model 和 temperature 参数
    llm = get_llm(model, temperature=temperature)
    
    #callback_manager 是一个包含 StdOutCallbackHandler 的列表，用于处理回调。
    callback_manager = [StdOutCallbackHandler()]

    #获取一个空操作的回调管理器实例，通常用于占位符。
    run_manager = CallbackManagerForChainRun.get_noop_manager()

    #使用从 get_llm 获取的语言模型 llm 以及 verbose 和 search_internet 参数创建 ChatMOF 实例。
    chatmof = ChatMOF.from_llm(
        llm=llm, 
        verbose=verbose, 
        search_internet=search_internet,
    )

    #打印欢迎信息和提示用户输入问题。
    print ('#' * 50 + "\n")
    print ('Welcom to ChatMOF!')
    print ("\n" + "#"*10 + ' Question ' + "#"*30)
    print ('Please enter the question below >>')
    question = input()
    
    #调用 chatmof 的 run 方法，传入用户输入的问题和 callback_manager 列表，获取输出。
    output = chatmof.run(question, callbacks=callback_manager)

    #打印 ChatMOF 的输出结果。
    print ('\n')
    print ("#"*10 + ' Output ' + "#" * 30)
    print (output)
    print ('\n')
    print ('Thanks for using CHATMOF!')

    return output


if __name__ == '__main__':
    #创建一个 ArgumentParser 对象，用于处理命令行参数。
    parser = argparse.ArgumentParser()

    #将 add_argument 方法赋值给 add，简化后续的调用
    add = parser.add_argument

    '''
    这段代码通过 argparse 模块向命令行解析器添加了一个参数 --model（还有其别名 --model-name 和 -m），
    这个参数接受一个字符串类型的值，用于指定 OpenAI 模型的名称。
    如果用户没有提供这个参数，则默认使用 gpt-4。帮助信息提供了关于这个参数的详细说明，包括可接受的模型名称和默认值。
    '''
    add(
        '--model',
        '--model-name',
        '-m',
        type=str,
        help="OpenAI model name. model_name must be one of [gpt-4, gpt-3.5-turbo, gpt-3.5-turbo-16k]. (default: gpt-4)",
        default='gpt-4'
    )
    
    #添加 --temperature 和 -t 参数，类型为浮点数，默认值为 0.1，并提供帮助信息。
    add(
        '--temperature',
        '-t',
        help='Temperature of LLM models. The lower the temperature, the more accurate the answer, and the higher the temperature, the more variable the answer. (default: 0.1)',
        type=float,
        default=0.1,
    )

    #遍历 str_kwargs_names 字典，为每个键添加一个可选的字符串类型的命令行参数。
    #使用格式化字符串 (f-string) 动态生成参数名。例如，如果 key 的值是 'example'，则生成的参数名是 --example。
    #这段代码动态地为命令行解析器添加了多个可选的字符串参数，参数名称和帮助信息来自于字典 str_kwargs_names。
    for key, value in str_kwargs_names.items():
        parser.add_argument(f"--{key}", type=str, required=False, help=f"(optional) {value}")

    for key, value in int_kwargs_names.items():
        parser.add_argument(f"--{key}", type=int, required=False, help=f"(optional) {value}")

    for key, value in float_kwargs_names.items():
        parser.add_argument(f"--{key}", type=float, required=False, help=f"(optional) {value}")

    #遍历 bool_kwargs_names 字典，为每个键添加一个可选的布尔类型的命令行参数，使用 store_true 动作。
    #使用 action='store_true' 的命令行参数是一种处理布尔选项或标志的简便方法。当参数在命令行中提供时，其值为 True；当参数不在命令行中提供时，其值为 False。
    for key, value in bool_kwargs_names.items():
        parser.add_argument(f"--{key}", action='store_true', required=False, help=f"(optional) {value}")

    #解析所有的命令行参数。
    args = parser.parse_args()

    #获取 model 参数的值。
    model = args.model
    #获取 temperature 参数的值。
    temperature = args.temperature

    #创建一个空字典 kwargs，用于存储其他可选参数。
    kwargs = {}

    #这段代码的功能是遍历四个字典中的所有键，检查每个键是否在命令行参数中提供，如果提供了则将其添加到一个新的字典 kwargs 中
    #四个字典的键合并成一个迭代器。遍历四个字典中的所有键
    for key in chain(str_kwargs_names.keys(), 
                        int_kwargs_names.keys(),
                        float_kwargs_names.keys(),
                        bool_kwargs_names.keys(),
                        ):
        #返回命令行中提供的该键对应的值，如果没有提供该键，则返回 None。
        #这是一个赋值表达式（海象运算符 :=），用于在条件语句中赋值。
        if value := getattr(args, key):
            #如果上一步中的 value 不为 None 或 False，则将 key 和 value 添加到字典 kwargs 中。
            kwargs[key] = value

    #调用 main 函数，传入 model 和 temperature 参数，以及其他可选参数。
    main(model=model, temperature=temperature, **kwargs)
