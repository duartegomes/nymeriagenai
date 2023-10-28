from PIL import Image
import pandas as pd
import os
import openai
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import streamlit as st
from utils import get_completion_from_messages, generate_sales_email, feature_engineering, contact_information


openai.api_type = "azure"
openai.api_base = "https://jtaopenai.openai.azure.com/"
openai.api_version = "2023-08-01-preview"
openai.api_key = "2678a8ae8aa94770b674ebcc5fe3ce20"

image = Image.open('data/jta.png')
st.image(image, width=100)


def ask_company() -> None:
    """
    Handles the company selection and goal setting.
    """
    st.session_state.external = pd.read_csv(
        "data/external.csv", encoding='cp1252').drop("Unnamed: 0", axis=1)
    st.session_state.internal = pd.read_csv(
        "data/internal.csv", encoding='cp1252').drop("Unnamed: 0", axis=1)
    st.session_state.nymeria = pd.read_csv(
        "data/nymeria.csv", encoding='cp1252').drop("Unnamed: 0", axis=1)
    st.session_state.engagement = pd.read_csv(
        "data/contacts.csv", encoding='cp1252').drop("Unnamed: 0", axis=1)

    numeric_columns = st.session_state.internal.select_dtypes(
        include=['number']).columns
    st.session_state.internal[numeric_columns] = st.session_state.internal[numeric_columns].round(
        2)

    target = [" "]
    target += list(st.session_state.external["Company Name"].values)

    st.session_state.target = st.selectbox(
        "What Company are you interested in?", target, key='cmp_selection')

    if st.session_state.target != " ":
        st.session_state.goal = st.selectbox("What do you need help with?", [
                                             " ", "Information Aggregator", "Email Generator"], key='goal_selection')

        if st.session_state.goal == "Email Generator":
            st.session_state.ma = st.selectbox("Which should be the Main Focus?", [
                                               "Azure", "Modern Work"], key='main_focus')
            st.session_state.at = st.text_input(
                "Who should be the sender?", "", key='agent')
            st.session_state.sp = st.text_input(
                "Any Main Selling Point? (Optional)", "", key='selling_point')

    st.session_state.additional_details = st.button(
        'Submit', key='general_submit')

    if (((st.session_state.target == " ") or (st.session_state.goal == " ")) and (st.session_state.additional_details)):
        st.warning('Please fill all the mandatory fields.', icon="⚠️")
    else:
        if st.session_state.additional_details:
            if st.session_state.goal == "Email Generator":
                st.session_state.writte_email = True
                st.experimental_rerun()
            elif st.session_state.goal == "Information Aggregator":
                st.session_state.ask_company = True
                st.experimental_rerun()


def chat_info():
    """
    Displays external, internal, and Nymeria engine information.
    """
    ext_info = st.session_state.external[st.session_state.external["Company Name"]
                                         == st.session_state.target]
    int_info = st.session_state.internal[st.session_state.internal["CustomerTPName"]
                                         == st.session_state.target]
    nymeria = st.session_state.nymeria[st.session_state.nymeria["Company Name"]
                                       == st.session_state.target]
    eng_info = st.session_state.engagement[st.session_state.engagement["Company Name"]
                                           == st.session_state.target]
    eng_info.fillna("No record found", inplace=True)
    eng_info = eng_info.apply(contact_information, axis=1).values[0]
    int_info_summary = int_info.apply(feature_engineering, axis=1)

    st.warning("The data represented here, while mostly real, is not 100% accurate. This is intended for internal testing.", icon="⚠️")

    messages = [{'role': "system", "content": "You are a business summarization bot, your main goal is to present a business summary. Always, if available, present the website, the industry, the size and the holding. Remove subjective information. Always use bullet points"},
                {"role": "user", "content":  "Talk briefly about this company: " + str(ext_info.iloc[:, 1:])}]
    response1 = get_completion_from_messages(messages)

    with st.expander("External Information"):
        st.write(response1)

    messages = [{'role': "system", "content": "You are a business re-write bot, your main goal is present information in text. Always use bullet points. Separate information into three sections: one related with company demographics, the other with what products are purchased and the other with purchase pattern (consumption and changes)."},
                {"role": "user", "content":  "Talk about this company: " + str(int_info_summary.to_dict())}]
    response2 = get_completion_from_messages(messages)
    with st.expander("Internal Information"):
        st.write(response2)

    with st.expander("Previous Engagements Information"):
        st.write(eng_info)

    with st.expander("Nymeria Enhanced Information"):
        st.write("According to the Nymeria Engine " + nymeria["Company Name"].values[0] +
                 " presents: " + "\n - " + nymeria["Reccomendation"].values[0])

    messages = [{'role': "system", "content": "If there is any relevant difference between internal and external information, present it as bullet points."},
                {"role": "user", "content":  "external source: " + str(ext_info[["Industry", "Company size"]].to_dict()) + "and the internal source " + str(int_info[["Industry", "Organization Size"]].rename({"Organization Size": "Company size"}, axis=1).to_dict())}]
    response2 = get_completion_from_messages(messages)
    with st.expander("Conflicting Information"):
        st.write(response2)


def chat_email():
    """
    Generates and displays a sales email based on selected parameters.
    """
    ext_info = st.session_state.external[st.session_state.external["Company Name"]
                                         == st.session_state.target]
    int_info = st.session_state.internal[st.session_state.internal["CustomerTPName"]
                                         == st.session_state.target]

    st.warning("The data represented here, while mostly real, is not 100% accurate. This is intended for internal testing.", icon="⚠️")

    messages = [{'role': "system", "content": f"You are intended to write emails as a Sales Agent from Microsoft. You will write the email in a friendly tone, being very detailed in helping the customer. Always contextualize the email in light of  the industry of the customer, which is {ext_info['Industry'].values[0]}. Lastly, remember to mention that you will be calling over the upcoming days. Never offer promotions or discounts. "},
                {"role": "user", "content": generate_sales_email(int_info, ext_info, st.session_state.ma)}]

    response_email = get_completion_from_messages(messages)
    st.write(response_email)


def launch_assistant() -> None:
    """
    Main function to launch the assistant based on user interaction.
    """
    if 'ask_company' not in st.session_state:
        st.session_state.ask_company = False

    if 'writte_email' not in st.session_state:
        st.session_state.writte_email = False

    if st.session_state.ask_company:
        chat_info()
    elif st.session_state.writte_email:
        chat_email()
    else:
        ask_company()


if __name__ == '__main__':
    launch_assistant()
