import pandas as pd
import os
import openai
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import streamlit as st


def get_completion_from_messages(messages,
                                 model="gpt-35-turbo",
                                 temperature=0,
                                 max_tokens=1000):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        engine="test"
    )
    return response.choices[0].message["content"]


def get_yearly_change_message(product, yearly_change):
    if yearly_change < 0:
        return f"Unfortunately, they have decreased consumption by {abs(yearly_change)}% over the last year."
    elif yearly_change > 0:
        return f"We are glad to see that they have increased consumption by {yearly_change}% over the last year."
    elif pd.isna(yearly_change):
        return f"We are glad to see that you have started to use our product over the last year."
    else:
        print(yearly_change, yearly_change,
              yearly_change, yearly_change, yearly_change)
        return "Their consumption has remained unchanged over the last year."


def generate_sales_email(internal_row, external_row, product):
    azure_consumption_column = 'Previous12_Azure'
    modern_work_consumption_column = 'Previous12_ModernWork'
    azure_yearly_change_column = 'YoY_Azure'
    modern_work_yearly_change_column = 'YoY_ModernWork'

    def get_azure_message():
        if internal_row[azure_consumption_column].values[0] != 0 and internal_row[azure_consumption_column].values[0] is not None:
            yearly_change_azure = internal_row[azure_yearly_change_column].values[0] * 100
            return f"The client is already using Azure. We would like to discuss how we can help {external_row['Company Name'].values[0]} further with Azure. {get_yearly_change_message('Azure', yearly_change_azure)}"
        elif internal_row[modern_work_consumption_column].values[0] != 0 and internal_row[modern_work_consumption_column].values[0] is not None:
            return f"The client is already using Microsoft's Modern Work Solution. Considering adding Azure to your suite of products can bring additional benefits, allowing seamless integration between Azure and Modern Work. Let's explore how this integration can enhance your organization's productivity and efficiency, particularly in the industry of {external_row['Industry'].values[0]}."
        else:
            return f"The client is not using Azure. We would like to introduce them to Azure and explain how we can help {external_row['Company Name'].values[0]} further with Azure. Let's explore opportunities for them to use more Azure, adapting to the client industry: {external_row['Industry'].values[0]}."

    def get_modern_work_message():
        if internal_row[modern_work_consumption_column].values[0] != 0 and internal_row[modern_work_consumption_column].values[0] is not None:
            yearly_change_modern_work = internal_row[modern_work_yearly_change_column].values[0] * 100
            return f"The client is already using Microsoft's Modern Work Solution. We would like to discuss how we can help {external_row['Company Name'].values[0]} further with Microsoft's Modern Work Solution. {get_yearly_change_message('Modern Work', yearly_change_modern_work)}"
        elif internal_row[azure_consumption_column].values[0] != 0 and internal_row[azure_consumption_column].values[0] is not None:
            return f"The client is already using Azure. Considering adding Microsoft's Modern Work Solution to your suite of products can bring additional benefits, allowing seamless integration between Azure and Modern Work. Let's explore how this integration can enhance your organization's productivity and collaboration, particularly in the industry of {external_row['Industry'].values[0]}."
        else:
            return f"The client is not using Microsoft's Modern Work Solution. We would like to introduce them to Microsoft's Modern Work Solution and explain how we can help {external_row['Company Name'].values[0]} further with Microsoft's Modern Work Solution. Let's explore opportunities for them to start using Microsoft's Modern Work Solution, adapting to the client industry: {external_row['Industry'].values[0]}."

    if product == 'Azure':
        return get_azure_message()
    elif product == 'Modern Work':
        return get_modern_work_message()
    else:
        return "Invalid product specified."


def contact_information(row):
    last_contact_type = row['Last Contact Type']
    last_contact_date = row['Last Contact Date']
    last_contact_agent = row['Last Contact Agent']
    placer = False
    contact_info = []

    if last_contact_type != "No record found":
        contact_info.append(f"Last contact was made by {last_contact_type}")
        placer = True
    else:
        contact_info.append("No info about the type of last contact")

    if last_contact_date != "No record found":
        contact_info.append(f"On {last_contact_date}")
        placer = True
    else:
        contact_info.append("No info about the date of last contact")

    if last_contact_agent != "No record found":
        contact_info.append(f"The agent responsible was {last_contact_agent}")
        placer = True
    else:
        contact_info.append(
            "No info about the agent who made the last contact")

    if placer == False:
        return "- No contact records available."
    else:
        return "\n".join([f"- {info}." for info in contact_info])


def feature_engineering(row):
    customer_info = f"Refers to {row['CustomerTPName']} from the {row['Subsidiary']} subsidiary and operate in the {row['Industry']} industry,  with Organization Size: {row['Organization Size']} and from  Cohort: {row['Cohort ']}."

    if pd.isnull(row['Previous12_TotalRevenue']) or row['Previous12_TotalRevenue'] == 0:
        return customer_info + "Not our  customer."
    else:
        current_consumption = row['Previous12_TotalRevenue']
        yearly_change = row['YoY_TotalRevenue']

        customer_status = customer_info + \
            f"Is a current customer. Last 12 months consumption value: {current_consumption}."
        if yearly_change != 0:
            customer_status += f" This represents an {'increase' if yearly_change > 0 else 'decrease'} of {abs(yearly_change*100)}% compared to the last 12 to 24 months."

        num_products_purchased = int(
            row['NProducts']) if not pd.isnull(row['NProducts']) else 0
        if num_products_purchased == 1:
            va = True
            product_info = ''
            if row['HasO365'] in [True, 'Yes']:
                product_info = "O365"
            elif row['Previous12_ModernWork'] not in [0, None]:
                product_info = "Microsoft Modern Work Solutions"
            elif row['Previous12_Azure'] not in [0, None]:
                product_info = "Azure"

            if product_info:
                customer_status += f" Only 1 product ({product_info})."
        else:
            va = False
            customer_status += f" Number of Products Purchased: {num_products_purchased}."

        if (row['HasO365'] in [True, 'Yes']) and (va is False):
            customer_status += " Consumes O365."

        if (row['Previous12_Azure'] not in [0, None]) and (va is False):
            azure_consumption_last_12_months = row['Previous12_Azure']
            customer_status += f" Azure customer. Azure Consumption in last 12 months: {azure_consumption_last_12_months}."

            azure_yearly_change = row['YoY_Azure'] if not pd.isnull(
                row['YoY_Azure']) else 0
            if azure_yearly_change != 0:
                customer_status += f" This represents an {'increase' if azure_yearly_change > 0 else 'decrease'} of {abs(azure_yearly_change*100)}% compared to the last 12 to 24 months."
        else:
            customer_status += " Not an Azure customer."

        if (row['Previous12_ModernWork'] not in [0, None]) and (va is False):
            modern_work_consumption_last_12_months = row['Previous12_ModernWork']
            customer_status += f" Modern Work customer. Modern Work Consumption in last 12 months: {modern_work_consumption_last_12_months}."

            modern_work_yearly_change = row['YoY_ModernWork'] if not pd.isnull(
                row['YoY_ModernWork']) else 0

            if modern_work_yearly_change != 0:
                customer_status += f" This represents an {'increase' if modern_work_yearly_change > 0 else 'decrease'} of {abs(modern_work_yearly_change*100)}% compared to the last 12 to 24 months."

        return customer_status
