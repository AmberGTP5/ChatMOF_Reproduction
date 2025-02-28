## This is a paper reproduction project.

### The following functions are implemented:
1. Add Chinese comments for main codes

2. The following problems were found during debugging:
- During the installation of the python library, the langchain library and the openai library versions may conflict.
- When calling the Chatgpt-3.5 interface, it will prompt that the balance is insufficient. Please bind a credit card. This may be caused by an abnormality caused by too high a calling frequency. You can try calling the gpt4 interface or calling a local large model.

![figure1](figures/fig1.jpg)

# ChatMOF : An Autonomous AI System for Predicting and Generating Metal-Organic Frameworks

ChatMOF is an autonomous Artificial Intelligence (AI) system that is built to predict and generate of metal-organic frameworks (MOFs). By leveraging a large-scale language model (GPT-4 and GPT-3.5-turbo), ChatMOF extracts key details from textual inputs and delivers appropriate responses, thus eliminating the necessity for rigid structured queries. The system is comprised of  three core components (i.e. an agent, a toolkit, and an evaluator) and it forms a robust pipeline that manages a variety of tasks, including data retrieval, property prediction, and structure generation. The study further explores the merits and constraints of using large language models (LLMs) AI system in material sciences using and showcases its transformative potential for future advancements.

NOTE: ChatMOF has been updated to 0.2.0 and includes a new tool `unit converter`.

## Online Demo
You can test ChatMOF in [Online-Demo](https://chatmof-online.streamlit.app/) site. It requires an `OpenAI key` (For more information, visit the [OpenAI site](https://platform.openai.com/docs/introduction)). In online demo, only the Search task is available. For prediction tasks and generation tasks that require machine learning do not work properly, except for the provided examples.

![OnlineDemo](figures/online_demo.JPG)

## Install

### Dependencies

NOTE: This package is primarily tested on Linux. We strongly recommend using Linux for the installation.

```
python>=3.9
```

### Installation

```bash
$ pip install chatmof
```
If you have a dependency issues between `torch` and `moftransformer`, uninstall `torch` and `pytorch-lightning`, install `torch <2.0.0`, and reinstall.

For prediction and generation task, you have to setup modules.

```bash
$ chatmof setup
```

Add the following line to `.bashrc` for the openai api key.
```shell
# openai api key
export OPENAI_API_KEY="enter_your_api_key"
```

If you want to search the internet, you'll need to enter the `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` into `.bashrc`.
```shell
# google api and cse_id
export GOOGLE_API_KEY="enter_your_api_key"
export GOOGLE_CSE_ID="enter_your_id"
```

For MOF generation, you need to install GRIDAY.

```bash
$ chatmof install-griday
```

## How to use ChatMOF
You can use it by running chatmof's `run` function.

```bash
$ chatmof run
```
![example](figures/example.JPG)

You can adjust argument of Chatmof such as model and temperature.

```bash
$ chatmof run --model-name gpt-3.5-turbo --temperature 0.5
```

You can use `help` to see more options.

```bash
$ chatmof run --help
```


## Example of ChatMOF
### 1) Search task
![ex1](figures/ex1.jpg)

### 2) prediction task
![ex2](figures/ex2.jpg)

### 3) prediction task
![ex3](figures/ex3.jpg)

## Architecture

ChatMOF comprises three core components: an agent, toolkits, and an evaluator. Upon receiving a query from human, the agent formulates a plan and selects a suitable toolkit. Subsequently, the toolkit generates outputs following the proposed plan, and the evaluator makes these results into a final response.

![figure1](figures/fig2.jpg)


## Citation
if you want to cite ChatMOF, please refer to the following paper:
> ChatMOF: An Autonomous AI System for Predicting and Generating Metal-Organic Frameworks, arxiv:2308.01423 [[link]](https://arxiv.org/abs/2308.01423)

## Contributing 🙌

Contributions are welcome! If you have any suggestions or find any issues, please open an issue or a pull request.

## License 📄

This project is licensed under the MIT License. See the `LICENSE` file for more information.

