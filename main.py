import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()


csv_file_path = "/home/amadgakkhar/code/UW Task/Autotrader_leads_inventory.csv"
memory = ConversationBufferMemory()


def query_data(query, agent):
    response = agent.invoke(query)
    return response


def read_csv() -> pd.DataFrame:
    csv_file_path = csv_file_path
    df_csv = pd.read_csv(csv_file_path)
    return df_csv


def get_agent():
    groq_api = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
        api_key=groq_api,
    )
    agent = create_csv_agent(
        llm,
        csv_file_path,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=None,
    )
    return agent


def main():
    agent = get_agent()

    st.title("Autotrader Leads Inventory")
    query = st.text_input("Query:")
    if st.button("Submit") and query:
        response = query_data(query, agent)
        st.write(response)


if __name__ == "__main__":
    main()
# while True:
#     query = input("Query: ")
#     if query == "exit":
#         break
#     agent = get_agent()
#     response = query_data(query, agent)
#     print(response)
