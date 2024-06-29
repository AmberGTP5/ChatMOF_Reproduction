# Structure of ChatMOF

## 根目录文件
- main.py : Functions for running ChatMOF # 这个文件包含运行ChatMOF的主要功能。它可能包括启动程序的入口点，以及控制程序流程的主要逻辑。
- config.py : Configuration of ChatMOF # 这个文件用于ChatMOF的配置设置，可能包括路径配置、参数设置以及其他全局配置项。
- utils.py : ChatMOF Utility Functions # 这个文件包含ChatMOF的通用实用函数，这些函数可能被项目中的多个模块使用。
- setup_module.py : Functions for setting up ChatMOF's module # 这个文件用于设置和初始化ChatMOF模块，可能包含安装依赖、初始化数据库等功能。

## 目录和子模块
- agents : Implement ChatMOF's agent and evaluator. # 这个目录实现了ChatMOF的代理和评估器。代理可能负责具体任务的执行，而评估器可能负责对任务结果进行评估。

- tools: Toolkit for ChatMOF # 这个目录包含ChatMOF的各种工具：
  - ase_repl : Tools for running ASE library # 运行ASE（Atomic Simulation Environment）库的工具。
  - genetic algorithm : Tools for generating MOF using genetic algorithm # 使用遗传算法生成MOF（Metal-Organic Framework）的工具。
  - predictor: Tools for predict MOF's properties using MOFTransformer # 使用MOFTransformer预测MOF属性的工具。
  - python_repl : Tools for running python code # 运行Python代码的工具。
  - search_csv : Tools for searching data from database # 从数据库中搜索数据的工具。
  - unit_converter: Tools for converting unit. # 单位转换工具。
  - visualizer: Tools for visualizing the MOF # 用于可视化MOF的工具。
  - tool_utils.py : Functions for integrating tools with ChatMOF # 用于将这些工具与ChatMOF集成的函数。

- llm : Large language models that work with ChatMOF # 这个目录包含与ChatMOF一起使用的大型语言模型：
  - codellama.py : Codellama, CodeLlama2 # CodeLlama和CodeLlama2模型的集成。
  - llama.py : Llama, Llama2 # Llama和Llama2模型的集成。
  - openai.py : GPT-4, GPT-3.5-turbo, GPT-3.5-turbo-16k # GPT-4, GPT-3.5-turbo, GPT-3.5-turbo-16k模型的集成。
  - utils.py : Functions for integrating LLMs with ChatMOF # 将这些LLM与ChatMOF集成的函数。

- Database: Database for using ChatMOF # 这个目录包含ChatMOF使用的数据库：
  - load_model : Fine-tuned MOFTransformer model # 已微调的MOFTransformer模型。
  - structures : MOF Structure File # MOF结构文件：
    - coremof : CoREMOF Structure File # CoREMOF结构文件。
    - generate : MOFs created using the Generator tool # 使用生成工具创建的MOF。
    - hMOF : hypothetical MOF (Used in generator tool) # 假设的MOF（用于生成工具）。
  - tables: The look-up table used by ChatMOF (used by the search_csv tool) # ChatMOF使用的查找表（由search_csv工具使用）。

## 命令行接口目录
- cli : Functions for the Commend-line interface # 这个目录包含与命令行接口相关的功能：
  - install_griday : Functions for installing GRIDAY # 安装GRIDAY的功能。
  - main.py : command line interface main function # 命令行接口的主要功能。
  - run.py : Functions for running ChatMOF # 运行ChatMOF的功能。
  - setup.py : Functions to install ChatMOF # 安装ChatMOF的功能
