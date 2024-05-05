# 2024 EESTech Hackathon

This repository focuses on experimenting the capacity of large language models (LLMs) in helping exploratory data analysis (EDA) of customer experience design (CX Design). 

In this project, [Falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct), [Flan](https://huggingface.co/google/flan-t5-xxl), and [GPT](https://openai.com/) was used.

## Idea Description

Our goal is to enhance the customer experience. To achieve the goal, the generative AI should identify pain points and evaluate product popularity by analyzing customer requests.


## Requirements

- [Python 3.6 or higher](https://www.python.org/downloads/)
- [LangChain library](https://python.langchain.com/en/latest/index.html)
- [Huggingface API key](https://huggingface.co/login?next=%2Fsettings%2Ftokens)
  


## Installation

#### 1. Clone the repository

```bash
git clone https://github.com/daveebbelaar/langchain-experiments.git
```

#### 2. Create a Python environment

Python 3.6 or higher using `venv` or `conda`. 

Using `venv`:

``` bash
cd ArgGenius
python3 -m venv env
source env/bin/activate
```

Using `conda`:
``` bash
cd ArgGenius
conda create -n langchain-env python=3.8
conda activate langchain-env
```

#### 3. Install the required dependencies
``` bash
pip install -e .
pip install -r requirements.txt
```

#### 4. Set up the keys in a .env file

First, create a `.env` file in the root directory of the project. Inside the file, add your OpenAI API key and Huggingface API key:

```makefile
OPENAI_API_KEY="your_api_key_here"
HUGGINGFACEHUB_API_TOKEN="your_api_key_here"
```

Save the file and close it. In your Python script or Jupyter notebook, load the `.env` file using the following code:
```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```

When needed, you can access the `HUGGINGFACEHUB_API_TOKEN ` as an environment variable:
```python
import os
api_key = os.environ['HUGGINGFACEHUB_API_TOKEN']
```

## Usage
Run locally on your local machine.
``` bash
streamlit run app.py
```

## Outcome

After executing the file locally, a standalone desktop app will be shown as follows:
(Please insert screenshot)
In the first field, we can choose the database between [Infineon Github Issues](https://github.com/Infineon) and [Infineon Developer Community](https://community.infineon.com/?profile.language=en). In the second field, we can select the corresponding repository name, which is the identical to the product name, to be analyzed. In the third field, we can determine the app to generate the analyzed result in terms of the Customer Experience Index (CX index).

#### Customer Experience Index (CX Index)

In this catagory, three aspects of the product: satisfaction, ease of use, and effectiveness, which follows the rule in [Customer Experience Index (CxPi)](https://www.satrixsolutions.com/blog/what-is-customer-experience-index-cxpi), are evaluated by the LLM model. The satisfaction score is displayed as Net Promoter Score (NPS), which is the result of the percentage of promoters (satisfactory) subtracted by the percentage of detractors (unsatisfactory). The score of ease of use is displayed as Customer Effort Score (CES), which is the result of the percentage of consent (easy) subtracted by the percentage of dissent (difficult). Similarly, the effectiveness score is the result of the percentage of consent (effective) subtracted by the percentage of dissent (ineffective). The customer experience index is then displayed, which is the sum of the three.



## Data Source

The data is confidentially provided by [Infineon Github Issues](https://github.com/Infineon) and [Infineon Developer Community](https://community.infineon.com/?profile.language=en)
