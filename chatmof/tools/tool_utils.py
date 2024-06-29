import warnings
from typing import Dict, Callable, List
from pydantic import ValidationError
from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.agents import load_tools, get_all_tool_names

from chatmof.tools.predictor import _get_predict_properties
from chatmof.tools.search_csv import _get_search_csv
from chatmof.tools.genetic_algorithm import _get_generator
from chatmof.tools.visualizer import _get_visualizer
from chatmof.tools.python_repl import _get_python_repl
from chatmof.tools.unit_converter import _get_unit_converter
from chatmof.tools.ase_repl import _get_ase_repl

#接受一个 BaseLanguageModel 实例和一个可选的 verbose 参数，返回一个 BaseTool 实例
#键是工具名称，值是生成相应工具实例的函数。
_MOF_TOOLS: Dict[str, Callable[[BaseLanguageModel], BaseTool]] = {
    "search_csv" : _get_search_csv,
    "predictor": _get_predict_properties,
    'generator': _get_generator,
    "visualizer": _get_visualizer,
    "python_repl": _get_python_repl,
    'ase_repl': _get_ase_repl,
    "unit_converter": _get_unit_converter,
}

#_load_tool_names 列表包含基本工具的名称。
_load_tool_names: List[str] = [
    #'python_repl',
    #'requests',
    #'requests_get',
    #'requests_post',
    #'requests_patch',
    #'requests_put',
    #'requests_delete',
    #'terminal',
    #'human',
    # 'llm-math',
    #'pal-math'
]

_load_internet_tool_names: list[str] = [
    'google-search',
    'google-search-results-json',
    'wikipedia',
]

#加载基本工具、自定义工具和（可选的）互联网工具。
#使用 _MOF_TOOLS 字典中的生成函数创建自定义工具实例。
def load_chatmof_tools(
        llm: BaseLanguageModel, 
        verbose: bool=False,
        search_internet: bool=True,    
    ) -> List[BaseTool]:

    #load_tools 函数根据传入的工具名称列表和语言模型实例加载工具
    tools = load_tools(_load_tool_names, llm=llm)
    custom_tools = [model(llm=llm, verbose=verbose) for model in _MOF_TOOLS.values()]

    if search_internet:
        try:
            internet_tools = load_tools(_load_internet_tool_names, llm=llm)
        except Exception as e:
            print(f'Warning: The internet tool does not work - {str(e)}')
            internet_tools = []
        return custom_tools + tools + internet_tools
    else:
        return custom_tools + tools
