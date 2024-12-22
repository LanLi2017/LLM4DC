## AutoDCWorkflow🔗: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark 

Code for paper [AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark](https://arxiv.org/abs/2412.06724). 

### Introduction
We investigate the reasoning capability of large language models (LLMs) for automatically generating data-cleaning workflows. To evaluate LLMs' ability to complete data-cleaning tasks, we implemented a pipeline for LLM-based <strong>Auto</strong> <strong>D</strong>ata <strong>C</strong>leaning <strong>Workflow</strong>, prompting LLMs on data cleaning operations to repair three types of data quality issues: duplicates, missing values, and inconsistent data formats. 

<img src="pics/autodcwf.png" align="middle" width="100%">
Figure. Architecture of AutoDCWorkflow framework. Given a raw table T that requires cleaning and a well-defined data analysis purpose P, our framework, AutoDCWorkflow outputs a minimal and clean table $T_N$ that is sufficient to address P, along with a complete data cleaning workflow $W_N$ consisting of a sequence of applied operations. This planning process involves three LLM agents: (1). <em>Select Target Columns<\em>, (2). <em>Inspect Column Quality<\em>, (3). <em>Generate Operation & Arguments<\em>. After predicting the next operation and its arguments, the framework sends a request to the OpenRefine API to apply the operation, resulting in an intermediate table. The revised target column in undergoes another quality inspection, and the iteration continues until the column meets the quality standards.


## Benchmark Description


## Dependencies [TODO]
To establish the environment run this code in the shell:
```bash
conda env create -f 
pip install 
```
That will create the environment `autodc` we used.




### Repo/Directories Introduction 





## Usage
### Environment setup [TODO]
Activate the environment by running
``````shell
conda activate 
``````

### Run [TODO]
Check out commands in `.py`

## Citation
If you find our work helpful, please cite as
```
@article{li2024autodcworkflow,
  title={AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark},
  author={Li, Lan and Fang, Liri and Torvik, Vetle I},
  journal={arXiv preprint arXiv:2412.06724},
  year={2024}
}
```

