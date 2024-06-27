import gradio as gr  
import requests 
import Backend
import json
API_URL = "http://127.0.0.1:8000/credit-analysis"  

# Function to call backend API
def call_credit_analysis_api(user_query, history):
    payload = {
        "user_query": user_query,
        "history": history
    }
    if history and history[0][0] is None:
        history[0][0] = user_query
    response = requests.post(API_URL, json=payload)
    response_content = response.json()
    if response.status_code == 200:
        return str(response_content['analysis_results'])
    else:
        return 'Error: Unable to process your request'
    
gr.ChatInterface(fn= call_credit_analysis_api, theme= gr.themes.Soft(), chatbot=
gr.Chatbot(layout=
'bubble'
,likeable= True, show_label= True, show_copy_button=True, value=[(None,
"Welcome. Ask me the users credit related questions."
)],)).launch()


