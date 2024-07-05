import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from pandasai.llm.openai import OpenAI

# Define the customized Streamlit output
class StreamlitOutput(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
    def format_dataframe(self, result):  
        st.dataframe(result["value"])
        return
    def format_plot(self, result):
        st.image(result["value"])
        return
    def format_other(self, result):
        st.write(result["value"])
        return


# Streamlit app configuration
st.set_page_config(layout='wide')
st.title("Prompt Based Data Analysis and Visualization")
st.markdown('---')

# File uploader using Streamlit
upload_csv_file = st.file_uploader("Upload Your CSV file for data analysis and visualization", type=["csv"])
if upload_csv_file is not None:
    data = pd.read_csv(upload_csv_file, encoding="latin-1")
    st.dataframe(data.head(5))
    st.write('Data Uploaded Successfully!')

st.markdown('---')
st.write('### Enter Your Analysis or Visualization Request')
llm = OpenAI(api_token='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')  # Replace with your API key



query = st.text_area("Enter your prompt")

    # Process the query if submitted
if st.button("Submit Query"):
        if query:
            st.write('### OUTPUT : ')
            st.markdown('---')
            
            # Train the model
            instruction = """
            You are an useful BI bot so don't hallucinate
            **If the user asks out of context questions then answer " This question is out of context"**.
            **If the user uses cuss words then answer  " Such Derogatory words are not entertained "**.
            **There are many accounts with their projects having a start date and an end date. If a user input is on active projects, you need to understand that active projects are those opportunities that have their current date falling between their start date and end date. that is active projects = OPPORTUNITY NAME:START DATE≤Current Date≤END DATE?"""

            # Add a spinner to indicate output generation
            with st.spinner("Generating Output..."):
                query_engine = SmartDataframe(data, config={'llm': llm, "response_parser": StreamlitOutput})
                concatenated_query = f"{query}\n{instruction}"
                answer = query_engine.chat(concatenated_query)
                st.write(answer)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
        else:
            st.warning("Please provide a prompt for analysis.")
