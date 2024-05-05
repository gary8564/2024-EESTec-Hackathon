import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st 
from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain.chains import SequentialChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
import openai
import torch
#import intel_extension_for_pytorch as ipex
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from getpass import getpass
import textwrap
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
import re
import time
from PIL import Image
import base64
from src.config import *
from src.session_state import session
from src.style.sidebar import render_sidebar 
from src.callbacks import clicked
from src.utils import load_data, store_data, vader_sentiment_scores
from src.prompt_templates import PromptTemplates
# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title=PAGE_TITLE, page_icon=Image.open(PAGE_ICON))
css_file = "./src/style/style.css"

# Design style
with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
# Initialize prompt template object
promptTemplates = PromptTemplates()


@st.cache_data(show_spinner=False)
def load_data(filepath):
    df = pd.read_pickle(filepath)
    df = df.loc[df['pr'] == 'issue']
    return df

@st.cache_resource(show_spinner=False)
def load_llm_model(temperature, model="falcon", max_new_token=500):
    assert model in ["gpt", "falcon", "flan", "llama"]
    # Load the HuggingFaceHub API token from the .env file
    load_dotenv(find_dotenv())
    HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    # Load the LLM model from the HuggingFaceHub
    repo_id_1 = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    repo_id_2 = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    falcon = HuggingFaceEndpoint(repo_id=repo_id_1, 
                                 temperature=temperature, max_new_tokens=500)
    flan = HuggingFaceEndpoint(repo_id=repo_id_2, 
                               temperature=temperature, max_new_tokens=500)
    gpt = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=temperature)
    #llama = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    #tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    #qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(weight_dtype=torch.quint4x2, # or torch.qint8
    #                                                                  lowp_mode=ipex.quantization.WoqLowpMode.NONE, # or FP16, BF16, INT8
    #                                                                  )
    #checkpoint = None # optionally load int4 or int8 checkpoint
    #llama_ipex = ipex.llm.optimize(llama, quantization_config=qconfig, low_precision_checkpoint=checkpoint) # model optimization and quantization
    #del llama
    llm = gpt 
    if model == "falcon":
        llm = falcon
    elif model == "flan":
        llm = flan
    #else:
    #    llm = llama_ipex   
    return llm

def agi_painPoints(title, body):
    loader = WebBaseLoader("https://www.zendesk.de/blog/customer-pain-points/")
    docs = loader.load()
    llm = load_llm_model(temperature=0.1)
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.run(docs)
    first_prompt_template = promptTemplates.create_identifyPainPoints_template()
    first_prompt = PromptTemplate(
                    input_variables=["title", "body", "summary"],
                    template=first_prompt_template,
                ) 
    identify_chain = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="identifiedPainPoints"
                    )
    second_prompt_template = promptTemplates.create_resolvePainPoints_template() 
    second_prompt = PromptTemplate(
                    input_variables=["identifiedPainPoints", "summary"],
                    template=second_prompt_template,
                )
    reolve_chain = LLMChain(llm=llm, prompt=second_prompt, output_key="advice")
    overall_chain = SequentialChain(chains=[identify_chain, reolve_chain], 
                                    input_variables=["title", "body", "summary"],
                                    output_variables=["identifiedPainPoints", "advice"],
                                    verbose=True)
    response = overall_chain({
                    'title': title,
                    'body': body,
                    'summary': summary
                    })
    painPoints = response["identifiedPainPoints"]
    advice = response["advice"]
    wrapped_text_painPoints = textwrap.fill(painPoints, width=100, break_long_words=False, replace_whitespace=False)
    wrapped_text_advice = textwrap.fill(advice, width=100, break_long_words=False, replace_whitespace=False)
    return wrapped_text_painPoints, wrapped_text_advice

def get_sentiment_analysis(title, body):
    prompt_template = promptTemplates.create_sentimentAnalysis_template()
    prompt = PromptTemplate(
                    input_variables=["title", "body"],
                    template=prompt_template,
                )
    llm = load_llm_model(temperature=0.5)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run({
                    'title': title,
                    'body': body
                    })
    print(response)
    llm_sentiment_score = 0
    if 'Positive' in response:
        llm_sentiment_score = 1
    elif 'Negative' in response:
        llm_sentiment_score = -1
    vader_sentiment_score = vader_sentiment_scores(f"{title}\n{body}")
    return (llm_sentiment_score + vader_sentiment_score) / 2

def agi_cesIndex(title, body):
    prompt_template = promptTemplates.create_cesIndex_template()
    prompt = PromptTemplate(
                    input_variables=["title", "body"],
                    template=prompt_template,
                )
    llm = load_llm_model(temperature=0.5)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run({
                    'title': title,
                    'body': body
                    })
    print(response)
    if '1' in response:
        return 1
    elif '2' in response:
        return 2
    elif '3' in response:
        return 3
    elif '4' in response:
        return 4
    elif '5' in response:
        return 5
    else:
        return None

def agi_cxIndex(title, body):
    prompt_template = promptTemplates.create_CxPi_template()
    prompt = PromptTemplate(
                    input_variables=["title", "body"],
                    template=prompt_template,
                )
    llm = load_llm_model(temperature=0.5)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run({
                    'title': title,
                    'body': body
                    })
    print(response)
    if '1' in response:
        return 1
    elif '2' in response:
        return 2
    elif '3' in response:
        return 3
    elif '4' in response:
        return 4
    elif '5' in response:
        return 5
    else:
        return None


@st.cache_data(show_spinner=False)
def design_page():
    with st.sidebar:
        render_sidebar()
    
    st.image(TITLE_IMAGE)  # TITLE and Creator information
    with st.chat_message("assistant"):
        st.write("Hi, I'm Daniel, the assistant of the customer experience design. I'm happly to help you to quickly get insights from the customer feedbacks.")  # add spacing

def main_page():
    # Initialize states
    session.init("status", 0)
    session.init("users", list())
    session.init("painPoints", list())
    session.init("resolution", list())
    session.init("sentiments", list())
    session.init("difficulties", list())
    session.init("effectiveness", list())
    session.init("ces", 0)
    session.init("nps", 0)
    session.init("cx_index", 0)
    design_page()
    input_data_source = st.radio(
            label="Select data source you would like to analyze",
            options=("Github", "Community forum"),
            horizontal=True,
            key="data",
            on_change=clicked,
            args=(0,)
        )
    
    if "Github" in input_data_source:
        df = load_data(DATA_FILEPATH)
        repo_names = df['repo_name'].unique().tolist()
        input_repo_name = st.selectbox('Select repository name',
                                repo_names,
                                index=0,
                                key="repo_name",
                                on_change=clicked,
                                args=(0,))
        input_analysis_options = st.radio(label="Select your analysis target",
                                         options=("Pain Points", "CX Index"),
                                         horizontal=True,
                                         key="analysis_option",
                                         on_change=clicked,
                                         args=(0,))
        st.write("\n")  # add spacing
        repo_name = session.get('repo_name')
        analysis_option = session.get('analysis_option')
        if os.path.isfile(SAVE_FILEPATH):
            results = load_data()
            session.update("results", results)
        st.button('Generate', on_click=clicked, args=(1,))
        if session.get("status") > 0:
            if session.get("status") == 1:
                input_contents = []  
                if (input_repo_name != "") and (input_analysis_options != ""):
                    input_contents.append(str(input_repo_name))
                    input_contents.append(str(input_analysis_options))
                if (len(input_contents) < 2):  # remind user to provide data 
                    st.error('Please select all required fields!') 
                else:  # initiate gen process
                    df = df[df['repo_name']==str(input_repo_name)]
                    if str(input_analysis_options) == "Pain Points":
                        painPoint_list = []
                        advice_list = []
                        with st.spinner('Processing...'):
                            for i in range(df.shape[0]):
                                painPoints, advice = agi_painPoints(df['title'].iloc[i], df['body'].iloc[i])
                                painPoint_list.append(painPoints)
                                advice_list.append(advice)
                        session.update("painPoints", painPoint_list)
                        session.update("users", df["user_login"].tolist())
                        session.update("resolution", painPoint_list)
                        st.write('\n')  # add spacing
                        subheader = st.empty()
                        subheader.subheader('Here are the list of all possible pain points from different Github users:\n')
                        time.sleep(0.02)
                        df_results = pd.DataFrame({
                                            "User name": df["user_login"].tolist(),
                                            "Pain Points": painPoint_list,
                                            "Advice": advice_list
                                        })
                        st.dataframe(df_results, use_container_width=True, hide_index=True)
                    else:
                        sentiment_list = []
                        ces_list = []
                        effective_list = []
                        neg_count = 0
                        pos_count = 0
                        nt_count = 0
                        vdiff_count =  0
                        diff_count = 0
                        mid_count = 0
                        ez_count = 0
                        vez_count = 0
                        eff_count = 0
                        neff_count = 0
                        with st.spinner('Processing...'):
                            for i in range(df.shape[0]):
                                sentiment_score = get_sentiment_analysis(df['title'].iloc[i], df['body'].iloc[i])
                                if sentiment_score == -1:
                                    neg_count += 1
                                    sentiment_list.append(random.randint(1, 2))
                                elif sentiment_score == 1:
                                    pos_count += 1
                                    sentiment_list.append(random.randint(4, 5))
                                else:
                                    nt_count += 1
                                    sentiment_list.append(3)
                                ces = agi_cesIndex(df['title'].iloc[i], df['body'].iloc[i])
                                effective = agi_cxIndex(df['title'].iloc[i], df['body'].iloc[i])
                                if ces == 5:
                                    vez_count += 1
                                elif ces == 4:
                                    ez_count += 1
                                elif ces == 3:
                                    mid_count += 1
                                elif ces == 2:
                                    diff_count += 1
                                else:
                                    vdiff_count += 1
                                ces_list.append(ces)
                                if effective == 5:
                                    eff_count += 1
                                elif effective == 4:
                                    eff_count += 1
                                elif effective == 2:
                                    neff_count += 1
                                elif effective == 0:
                                    neff_count += 1
                                effective_list.append(effective)
                        pos_rate = pos_count * 100 // df.shape[0]
                        neg_rate = neg_count * 100 // df.shape[0]
                        nps = pos_rate - neg_rate
                        diff_rate = (vdiff_count + diff_count) * 100 // df.shape[0]
                        ez_rate = (vez_count + ez_count) * 100 // df.shape[0]
                        overall_ces = ez_rate - diff_rate
                        cx_index = ((pos_count + vez_count + ez_count + eff_count) - (neg_count + vdiff_count + diff_count + neff_count)) * 100 // df.shape[0] 
                        session.update("sentiments", sentiment_list)
                        session.update("difficulties", ces_list)
                        session.update("effectiveness", effective_list)
                        session.update("nps", nps)
                        session.update("ces", overall_ces)
                        session.update("cx_index", cx_index)
                        plotly_colors = px.colors.qualitative.Plotly
                        labels = ['Positive','Neutral','Negative']
                        values = [pos_count, nt_count, neg_count]
                        colors = [plotly_colors[1], plotly_colors[-1], plotly_colors[2]]
                        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=colors))])
                        tab1, tab2, tab3 = st.tabs(["Net Promoter Score (NPS)", "Customer Effort Score (CES)", "CX Index"])
                        with tab1:
                            st.header("Net Promoter Score (NPS)")
                            txt = f"""
                                    <div>
                                        <span class='bold'> NPS = <span class='highlight green'>%PROMOTERS </span> - <span class='highlight red'>%DETRACTORS</span> = {nps}
                                        </span>
                                    </div>
                                   """
                            st.markdown(txt, unsafe_allow_html=True)
                            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                        with tab2:
                            st.header("Customer Effort Score (CES)")
                            df = pd.DataFrame([["Very Difficult", vdiff_count/df.shape[0]], ["Difficult", diff_count/df.shape[0]], ["Medium", mid_count/df.shape[0]], 
                                                ["Easy", ez_count/df.shape[0]], ["Very Easy", vez_count/df.shape[0]]], columns=["Difficulty of the product usage", "Percentage"])
                            fig = px.histogram(df, x="Difficulty of the product usage", y="Percentage", color="Difficulty of the product usage", 
                                                color_discrete_map = {'Very Difficult':'red', 'Difficult':'red', 'Medium':'yellow', 'Easy':'green', 'Very Easy': 'green'})
                            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                            difficulty = "medium"
                            if overall_ces > 0:
                                difficulty = "easy"
                            if overall_ces < 0:
                                difficulty = "difficult"
                            st.image(f'./image/{difficulty}.png')
                        with tab3:
                            st.header("CX Index")
                            st.subheader(f"Result:\n{cx_index}")
                            face = "neutralface"
                            if cx_index > 0:
                                face = "happyface"
                            if cx_index < 0:
                                face = "sadface"
                            st.image(f'./image/nps_{face}.png')
                saveButton = st.button(label="Save Result", on_click=clicked, args=(2,))
                if saveButton:
                    saving = st.toast('Saving result...')
                    new_sample = {
                                'Repository': [session.get('repo_name') for i in range(len(repo_names))],
                                'Pain Points': session.get('painPoints'),
                                'Resolution': session.get('resolution'),
                                'Sentiments': session.get('sentiments'),
                                'Difficulties': session.get('difficulties'),
                                'Effectiveness': session.get('effectiveness'),
                                'NPS': session.get('nps'),
                                'CES': session.get("ces"),
                                'CX Index': session.get("cx_index"),
                                }
                    if not session.has("results"):
                        session.init("results", [])
                    new_results = session.get("results").append(new_sample)
                    session.update("results", new_results)
                    store_data(session.get("results"))
                    time.sleep(1)
                    saving.toast('Saved!')

        st.button("Reset", key="reset", on_click=clicked, args=(0, True, session.get("results")))
    
    else:
        pass
        



    
                    
if __name__ == '__main__':
    # call main function
    main_page()
                
                

            