from typing import Any, Dict, List
from pydantic import BaseModel
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.agents import initialize_agent, AgentType
from chatmof.config import config
from chatmof.tools import load_chatmof_tools
from chatmof.agents.prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX

# 这个 ChatMOF 类是一个链式（chain）处理的框架，专门用于处理语言模型（LLM）生成的对话，
# 通过一个代理（agent）来执行查询操作
class ChatMOF(Chain):
    # 一个语言模型实例，用于生成对话内容。
    llm: BaseLanguageModel
    # 一个对象，用于执行具体的查询操作。
    agent: object
    # 一个布尔值，控制是否输出详细信息。
    verbose: bool = False
    # 分别是输入和输出的键，用于在字典中提取对应的值。
    input_key: str = "query"
    output_key: str = "output"
    
    #input_keys 和 output_keys 是两个只读属性，返回输入和输出的键名列表。
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    #_call 方法接收一个字典形式的 inputs，从中提取查询并通过 agent 执行，返回包含输出的字典。
    def _call(
            self,
            inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        query = inputs[self.input_key]
        output = self.agent.run(query)
        
        return {
            self.output_key: output
        }
    
    
    @classmethod
    # from_llm 类方法用于创建 ChatMOF 实例
    def from_llm(
        cls: BaseModel,
        llm: BaseLanguageModel,
        verbose: bool = False,
        search_internet: bool = True,
    ) -> Chain:
        
        # tools 是加载的一组工具，根据是否需要互联网搜索配置。
        tools = load_chatmof_tools(
            llm=llm, 
            verbose=verbose, 
            search_internet=search_internet
        )
        
        # agent_kwargs 是初始化代理所需的参数，包括前缀、格式指令和后缀。
        agent_kwargs = {
            'prefix': PREFIX,
            'format_instructions': FORMAT_INSTRUCTIONS,
            'suffix': SUFFIX
        }

        # initialize_agent 创建一个代理实例，配置各种参数。
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            agent_kwargs=agent_kwargs,
            handle_parsing_errors=config["handle_errors"],
        )

        # cls 返回一个新的 ChatMOF 实例，并传入代理和语言模型。
        return cls(agent=agent, llm=llm, verbose=verbose)
