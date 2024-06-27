import re
import sqlite3
import warnings
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from typing import List
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv() 


llm = openai.AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT_GPT").strip(),
        api_key=os.getenv("API_KEY_GPT").strip(),
        api_version=os.getenv("API_VERSION_GPT").strip()
)

def read_csv_to_dataframe():
    """
    Reads a CSV file and converts it into a DataFrame.

    Args:
    - file_path (str): The file path of the CSV file to be read.

    Returns:
    - df (pd.DataFrame): The DataFrame containing the data from the CSV file.
    """
    # Read the CSV file into a DataFrame
    file_path = get_file_path()
    df = pd.read_csv(file_path)

    return df

def get_file_path():
    # Specify the file path of the CSV file
    current_directory = os.path.dirname(os.path.realpath(__file__))
 
    # Define the absolute paths to the prompt files
    file_path = os.path.join(current_directory, 'updated_data.csv')

    return file_path

def connect_to_database(database_name):
    """
    Connect to the SQLite database.

    Parameters:
    - database_name: Name of the SQLite database.

    Returns:
    - Connection object and Cursor object.
    """
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    return conn, cursor

def create_loan_data_table(cursor):
    """
    Create a loan_data table in the SQLite database.

    Parameters:
    - cursor: Cursor object for executing SQL queries.
    """
    create_table_query = '''CREATE TABLE IF NOT EXISTS loan_data (
                            loan_id INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            date_of_birth DATE NOT NULL,
                            no_of_dependents INTEGER NOT NULL,
                            education TEXT NOT NULL,
                            self_employed TEXT NOT NULL,
                            income_annum INTEGER NOT NULL,
                            loan_amount INTEGER NOT NULL,
                            loan_term INTEGER NOT NULL,
                            cibil_score INTEGER NOT NULL,
                            residential_assets_value INTEGER NOT NULL,
                            commercial_assets_value INTEGER NOT NULL,
                            luxury_assets_value INTEGER NOT NULL,
                            bank_asset_value INTEGER NOT NULL,
                            loan_status TEXT NOT NULL,
                            CIBIL_rating TEXT NOT NULL,
                            income_level TEXT NOT NULL,
                            loan_rating TEXT NOT NULL,
                            loan_term_type TEXT NOT NULL,
                            dependent_level TEXT NOT NULL,
                            edu_status INTEGER NOT NULL,
                            type_employment INTEGER NOT NULL,
                            loan_approval INTEGER NOT NULL
                        );
                    '''
    cursor.execute(create_table_query)

def insert_data_into_table(conn, cursor, dataframe, table_name):
    """
    Insert data from a DataFrame into the specified table in the SQLite database.

    Parameters:
    - conn: Connection object to the database.
    - cursor: Cursor object for executing SQL queries.
    - dataframe: Pandas DataFrame containing the data to insert.
    - table_name: Name of the table to insert data into.
    """
    dataframe.to_sql(table_name, conn, if_exists='append', index=False)

    # Commit changes and close connection
    conn.commit()
    conn.close()

def generate_sql_query(user_input):
    sql_query_prompt = f"""
    ## Problem Statement
        As an expert in SQLite, your task is to assist users with database queries related to loan data by generating syntactically correct SQLite queries following specific guidelines.

        ## Input Question
        User input: {user_input}

        ## Guidelines for Query Generation
        1. In case the User input does not include a unique identifier- loan ID or any specific number, the model should respond with the message 'Please provide the loan ID of the user'.
        2. Query only name column and the other necessary columns from tables; avoid querying all columns, the name column must always be included.
        3. Use column names from the provided database schema and avoid querying non-existent columns.
        4. Consider loan_approval column: 0 for loan denial, 1 for loan approval. If loan_approval is 1, indicate loan can be given but do not say it's been approved.
        5. Only generate one SQL query in the response, it should be the most relevant SQL query.
        6. For User input not related to loan approval, banking, credit analysis return a response "Not able to generate query."
        7. Find the keywords in the User input which are similar to or match the loan data table's columns and include all of those columns in the resultant SQL query. Do this only if User input is related to loan approval, banking, credit analysis.
        8. Ensure that the loan ID mentioned in the user query is accurately reflected in the generated SQL query.
        9. If the User input specifies a specific number of examples, limit the query results using `LIMIT`.
        ## Database Schema
        ```sql
        CREATE TABLE loan_data (
        loan_id INT NOT NULL,
        name VARCHAR(255) NOT NULL,
        date_of_birth DATE NOT NULL,
        no_of_dependents INT NOT NULL,
        education VARCHAR(255) NOT NULL,
        self_employed VARCHAR(255) NOT NULL,
        income_annum INT NOT NULL,
        loan_amount INT NOT NULL,
        loan_term INT NOT NULL,
        cibil_score INT NOT NULL,
        residential_assets_value INT NOT NULL,
        commercial_assets_value INT NOT NULL,
        luxury_assets_value INT NOT NULL,
        bank_asset_value INT NOT NULL,
        loan_status VARCHAR(255) NOT NULL,
        CIBIL_rating VARCHAR(255) NOT NULL,
        income_level VARCHAR(255) NOT NULL,
        loan_rating VARCHAR(255) NOT NULL,
        loan_term_type VARCHAR(255) NOT NULL,
        dependent_level VARCHAR(255) NOT NULL,
        edu_status INT NOT NULL,
        type_employment INT NOT NULL,
        loan_approval INT NOT NULL,
        PRIMARY KEY (loan_id)
        );
        Response Format
        Provide only the generated single SQL query without any additional characters or phrases.
        Avoid including any text other than the SQL query.
        For queries that cannot be translated into SQL, return "Not able to generate query."
        Query only name column and the other necessary columns from tables; avoid querying all columns, the name column must always be included.
        Find the keywords in the User input which are similar to or match the loan data table's columns and include all of those columns in the resultant SQL query. Do this only if User input is related to loan approval, banking, credit analysis.
        In case the User input does not include a unique identifier such as a loan ID or any specific number, you  should respond with the message 'Please provide the loan ID of the user'.
        Task Examples:

        User: "Can loan be disbursed to loan id 3?"
        Response: "SELECT loan_approval, name FROM loan_data WHERE loan_id = 3;"

        User: "What is your name?"
        Response: "Not able to generate query."

        User: "What is cibil rating of loan id 34?"
        Response: "SELECT CIBIL_rating, name FROM loan_data WHERE loan_id = 34;"

        User: "Can loan be disbursed to loan id 5 based on their cibil rating?"
        Response: "SELECT loan_approval, CIBIL_rating, name FROM loan_data WHERE loan_id = 5;"

        User: "Can loan be given to Vikas?"
        Response: "Please provide the loan id of user"

        User: "Can loan be given to 1?"
        Response: "SELECT loan_approval, name FROM loan_data WHERE loan_id = 1;"

    """
    response = llm.chat.completions.create(
    model=os.getenv("MODEL_GPT").strip(),
    temperature=0.0,
    max_tokens=1000,
    top_p=1,
    messages=[
        {"role": "user", "content": sql_query_prompt}
    ],
    )
    sql_query = response.choices[0].message.content
    if sql_query=='' or sql_query is None or 'sql_query' =="Not able to generate query.":
        return "Sorry there is no data available in the database for your question. Please check your query again."
    sql_query_result = execute_sql_query(sql_query)
    print(f"sql_query_result: {sql_query_result}")
    return sql_query_result

def get_user_input():
    # user_input = input("Enter your query or question: ")
    user_input = "Can loan be disbursed to loan id 1?" #TODO make this dynamic
    return user_input

def execute_sql_query(sql_query):
    # Connect to the database and execute the SQL query
    db = SQLDatabase.from_uri("sqlite:///credit_score_loan_data.db")
    if sql_query == "Not able to generate query.":
        result = sql_query
    else:
        result = db.run(sql_query)
    print(f"Result from db {result}")
    return result

def process_query_result(result, user_query):
    print("In process_query_result")
    query_to_text_prompt = f"""

    ## Problem Statement:
    You are a helpful assistant chatbot that interacts with a database to answer user query. Given a sql query result {result} from the loan data table, and the user query {user_query} , 
    generate a human-readable answer of the information present in the sql query result, answering only the information relevant to the user query. 
    
    ## Input Data:
    The sql query result contains one of the follwoing column's data:
    - name: Name of the loan applicant
    - loan_id: Unique identifier for each loan
    - date_of_birth: Date of birth of the loan applicant
    - no_of_dependents: Number of dependents of the loan applicant
    - education: Education level of the loan applicant
    - self_employed: Employment status of the loan applicant
    - income_annum: Annual income of the loan applicant
    - loan_amount: Amount of loan applied for
    - loan_term: Term of the loan in months
    - cibil_score: Credit score of the loan applicant
    - residential_assets_value: Value of residential assets
    - commercial_assets_value: Value of commercial assets
    - luxury_assets_value: Value of luxury assets
    - bank_asset_value: Value of assets in bank
    - loan_status: Status of the loan application (approved or rejected)
    - CIBIL_rating: Credit rating of the loan applicant
    - income_level: Income level of the loan applicant
    - loan_rating: Rating of the loan application
    - loan_term_type: Type of loan term (short-term or long-term)
    - dependent_level: Level of dependence on loan
    - edu_status: Education status of the loan applicant
    - type_employment: Type of employment of the loan applicant
    - loan_approval or loa: Approval status of the loan (1 for approved, 0 for rejected) (It can also be called as loan status or loan approval status)
    
    ## Output Format:
    - Only give your response as output and nothing else.
    ## Guidelines:
    - Ensure your response is grammatically correct, concise, informative, and easy to understand.
    - Include key statistics and trends from the sql query result.
    - Use natural language and avoid technical jargon in your response. 
    - Ensure your response only contains a human readable answer and nothing more.
    - 1 indicates loan has been approved and 0 means loan has not been approved

    ## Task Examples:
    user query: "What is cibil rating of loan id 34?"
    sql query result: [('Excellent',)]
    your response : Cibil rating of loan id 34 is Excellent.
    
    user query: "Can loan be disbursed to loan id 5 based on their cibil rating?"
    sql query result: [(1, 'Excellent', 'Mrs. Yvonne Young')]
    your response : The cibil rating of Mrs. Yvonne Young is excellent based on that and other factors like their asset value and their bank rating, the loan can be approved to Mrs. Yvonne Young.

   
    user query: "Can loan be disbursed to loan id 2?"
    sql query result: [(0, 'Charles King')]
    your response: Loan cannot be disbursed to Charles King based on many factors lik etheir cibil rating, assets value, loan rating etc.

    user query : "What is loan status of loan id 2?"
    sql query result: [(0, 'Emily Johnson')]
    your response: Loan status of Emily Johnson with loan id 2 is Not approved.

    user query : "What is loan status of loan id 3?"
    sql query result: [(1, 'Jake Blind')]
    your response: Loan status of Jake Blind with loan id 3 is approved.



    ## Restrictions:
    - Ensure your response is grammatically correct, concise, informative, and easy to understand.
    - Include key statistics and trends from the sql query result.
    - Use natural language and avoid technical jargon in your response. 
    - Ensure your response only contains a human readable answer and nothing more.
    - 1 indicates loan has been approved and 0 means loan has not been approved

    
    """
    response = llm.chat.completions.create(
    model=os.getenv("MODEL_GPT").strip(),
    temperature=0.2,
    max_tokens=1000,
    top_p=1,
    messages=[
        {"role": "user", "content": query_to_text_prompt}
    ],
    )
    
    answer = response.choices[0].message.content
    print(f"Process query result response : {answer}")
    return answer

def is_follow_up_question(user_query, chat_history):
 
    '''
    This function decides whether the question asked by a user is a follow up question to the previous question or not by passing it to LLM.
 
    '''
    print(f"The original user query is: {user_query}")
    context_list = [f"{qa[0]}\n{qa[1]}" for qa in chat_history if qa[0] is not None]
    if len(context_list)==0: #User is asking his first question of the conversation.
        return user_query
   
    else:
        context_prev_q_a= ('\n').join(context_list)
    print(f"The previous question answer context: {context_prev_q_a}")
    prompt = f""""
 
   
    #Context:
    Your first task is to rephrase the current question (user query) into gramatically correct format. Post rephrasing, given a context list and a the rephrased current question, your task is to perform a semantic similarity to the previous question and determine if it is a follow-up to the previous questions or not.
    After this, you need to return a standalone version of the current question such that it can be answered individually.
    
    Output Format:
    Response: The model's response should be the standalone version of the follow-up question, or the current question if it is not a follow-up.
    
    Specifications:
    Task: Analyze the provided context, consisting of a previous question and its response, along with the current question, to determine if it is a follow-up question to the previous context.
    
    Instructions:
    1. Check if the user_query is grammatically and meaningfully correct. If it is not, then rephrase the user query such that it forms a grammatically correct question or statement.
    2. Determine if the current question is a follow-up to the previous conversation stored in the Context list.
    
    3. If the current question is a follow-up:
    - Generate a standalone version of the follow-up question based on the information from the previous question and its answer. 
    4. If the current question is not a follow-up, return the question as it is, treating it as a new question.
    5. The standalone question you respond with must include loan id.
    6. Avoid providing personal explanations for why a question is a follow-up; return the output directly.
    7. Ensure the standalone version of the follow-up question is clear and can be answered independently.
    8. The decision whether the current question is a follow-up or not has to be made very accurately. This is very important.
    
    Task Examples:
    
    Example 1:
    
    Current Question: "What is their cibil rating?"
    Decision to Rephrase: This question is grammatically correct so no need to rephrase.
    Previous Question Answer List: ["Question: What is the income level of loan id 1? \n Answer: The income level of Vikas with loan id 1 is high"]
    Decision: In the current question, 'Their' refers to Vikas with loan id 1. Hence the current question is a follow-up to the previous question and hence a standalone version of the current question must be generated.
    Response:
    What is the cibil rating of loan id 1?
    
    Example 2:

    Current Question: "What is the CIBIL rating for loan ID 2?"

    Decision to Rephrase: This question is grammatically correct so no need to rephrase.

    Previous Question Answer List: ["Question: What is the income level for loan ID 1? \n Answer: The income level for Vikas with loan ID 1 is high"]

    Decision: In the current question, the bank agent is asking a different question for a different user, about the CIBIL rating for another specific loan ID (loan ID 2). Hence it is a new question and not a follow-up question to the previous question.

    Response:
    What is the CIBIL rating for loan ID 2?


    Example 3:


    Current Question: "Can her loan be approved?"

    Decision to Rephrase: This question is grammatically correct so no need to rephrase.

    Previous Question Answer List:
    ["Question: What is the CIBIL rating for loan ID 2? \n Answer: The CIBIL rating of shalini with loan ID 2 is excellent",
    "Question: What is their loan term? \n Answer: The loan term of Shalini with loan id 2 is 8"]

    Decision: The current question is asking about loan approval where 'her' refers to shalini whose loan id is 2 it is a follow up question to a follow up question, which means it is a follow-up to the previous question and it's previous question. Hence, a standalone version of the current question must be generated.

    Response: Can loan be approved for loan ID 2?
    

    Example 4:

    Current Question: "What is the loan term for him?"

    Decision to Rephrase: This question is grammatically correct so no need to rephrase.

    Previous Question Answer List: ["Question: What is the loan approval status for loan ID 3? \n Answer: The loan approval status for John with loan ID 3 is Approved"]

    Decision: The bank agent is seeking information about the loan terms where "him" refers to John with loan Id 3 as seen in previous answer, which is a follow-up question related to the previous question about loan approval status. Hence, a standalone version of the current question must be generated.

    Response:
    What are the loan terms for loan ID 3?


    Response:
    Could you provide details about the loan approval status for loan ID 4?
    
    
    Additional Information:
    1. Rephrase the current question meaningfully and grammatically.
    3. Avoid providing personal interpretations or explanations.
    4. Base the response solely on the given context without speculation.
    5. Adhere strictly to the specified output format and instructions.
    
    Restrictions:
    1. Avoid providing personal explanations or interpretations for why a question is a follow-up.
    2. Return the output directly without additional commentary. You need to return only the standalone question and nothing else.
    3. Adhere strictly to the provided specifications and instructions.
    
    Here is the Current Question: {user_query}
    Here is the Context for the Current Question which includes the previous question and answers: {context_prev_q_a}
    
    
    
    """
    response = llm.chat.completions.create(
    model=os.getenv("MODEL_GPT").strip(),
    temperature=0.0,
    max_tokens=5000,
    top_p=1,
    messages=[
          {"role" : "user", "content" : prompt}
    ],
    )
    generated_response = response.choices[0].message.content

    print(f"The generated standalone question is : {generated_response}")
    return generated_response

def get_multiple_question(user_query) -> list[str]:
    print("In get multiple question")
    '''
    This function identifies whether a question is complex, and returns a list of simpler questions.
    '''
 
 
    prompt = f'''
    ## Context:
    Your task is to analyze the user's query and determine whether it can be split into simpler queries.
    Make a decision to see if the given User_query is complex or simple. If it is a complex question that can be split into sub-questions, please do.
    If the question is simple, let it be as it is.
 
    ## Specifications:
    Task: Analyze the user's query and determine whether it can be split into simpler queries.
 
    ## Instructions:
    1. If the user's query is complex, split it into subqueries and return only the subqueries.
    2. If the user's query is not complex, return the question as it is.
    3. Avoid providing explanations on how or why the question is being split.
    4. Return only the subqueries and nothing else.
 
    ## Task Examples:
 
    Example 1:
 
    Question: Compare the cibil rating of load id 1 and loan id 2.
    Decision: This query is asking the information of the cibil rating of two users with loan id 1 and loan id 2. Hence it is a complex question and it can be split into two simple question.
    Your Response:
    What is the cibil rating of loan id 1?
    What is the cibil rating of loan id 2?
 
    Example 2:
 
    Question: Compare the Cibil rating and Income level of Loan id 1 and loan id 2? Which user is better in terms of these two parameters?
    Decision: This query is asking the information of the Cibil rating and Income level of two users with load id 1 and loan id 2. Hence it is a complex question and it can be split into multiple simple questions.
   
    Your Response:
 
    What is Cibil rating of loan id 1?
    What is Income level of loand id 1?
    What is Cibil rating of loan id 2?
    What is Income level of loand id 2?
 
    ## Additional Information:
    1. Ensure that subqueries are generated accurately based on the complexity of the user's query.
    2. Avoid providing explanations or justifications for splitting the question.
    3. Base the response solely on the complexity of the user's query.
 
    ## Restrictions:
 
    1. Avoid providing explanations on how or why the question is being split.
    2. Strictly return only the simple sub-questions and nothing else.
 
    Here is the question to be split: {user_query}
   
'''
   
    response = llm.chat.completions.create(
    model=os.getenv("MODEL_GPT").strip(),
    messages=[
          {"role" : "user", "content" : prompt}
    ],
        temperature=0.0,
        max_tokens=4000,
        top_p=0.92,
    )
    generated_response = response.choices[0].message.content
    print(f"Generated response is {generated_response}")
    list_of_questions=re.split(r'[?.]', generated_response)
    print(f"List of functions: {list_of_questions}")
    for i,j in enumerate(list_of_questions):
        if j=='':
            del list_of_questions[i]
    punctuated_list = []
    for string in list_of_questions:
        string = string.replace("\n", "")
        if string.startswith(("What", "Where", "How", "Why", "When")):
            punctuated_list.append(string + "?")
        else:
            punctuated_list.append(string + ".")
    print(f"punctuated_list: {punctuated_list}")
    return punctuated_list

def generate_combined_summarized_response(all_the_llm_responses, user_query, punctuated_list):
    combined_response = ' '.join(all_the_llm_responses)
    print(combined_response)
    return combined_response


def execute_if_multiple_queries(punctuated_list, user_query):
    print("In execute_if_multiple_queries")
    if len(punctuated_list)>1: 
        #then there are muliple questions
        print("There are multiple questions")
        all_the_llm_responses = []
        for query in punctuated_list:
            sql_query_result = generate_sql_query(query)
            # sql_query_result = execute_sql_query(sql_query)
            llm_response = process_query_result(sql_query_result,query)
            all_the_llm_responses.append(llm_response)
        print(f"All llm response: {all_the_llm_responses}")
        response = generate_combined_summarized_response(all_the_llm_responses, user_query, punctuated_list)
    else: 
        sql_query_result = generate_sql_query(punctuated_list[0])  
        response = process_query_result(sql_query_result,user_query)
    return response

def is_greeting_only(user_input):
    # List of common greetings
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']

    # Regular expression to match greetings
    greeting_pattern = '|'.join([re.escape(greeting) for greeting in greetings])
    greeting_regex = re.compile(f'\\b({greeting_pattern})\\b', re.IGNORECASE)

    # Check if user input is only a greeting
    if greeting_regex.search(user_input):
        return True
    else:
        return False       
def generate_credit_analysis(user_query, history):
    if user_query.lower() in ["bye", "exit", "goodbye"]:
        return "Bye, Have a good day!"
    loan_data = read_csv_to_dataframe()
    # Connect to the database
    conn, cursor = connect_to_database('credit_score_loan_data.db')
    # Create the loan_data table
    create_loan_data_table(cursor)
    db = SQLDatabase.from_uri("sqlite:///credit_score_loan_data.db")
    if is_greeting_only(user_query):
        return "Hello there! I am here to help with answering the credit analysis questions of a user. How can I assist you today?"
    # substrings_to_check = ['id', 'loan_id', 'loan number', 'loan_number', 'loan id']
    # standalone_question = is_follow_up_question(user_query, history)
    # list_of_questions = get_multiple_question(standalone_question)
    list_of_questions = get_multiple_question(user_query)
    response = execute_if_multiple_queries(list_of_questions, user_query)
    print(f"Response: {response}")
    print(f"{list_of_questions}")
    return response

from fastapi import FastAPI, HTTPException, Query  
from pydantic import BaseModel  
import pandas as pd  
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(debug = True)  

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class AnalysisResponseModel(BaseModel):  
    user_query: str  
    history: List[List[str]]

@app.post("/credit-analysis")  
def credit_analysis_api(user_data : AnalysisResponseModel):  
    try:  
        print("user_query",user_data.user_query)
        analysis_results = generate_credit_analysis(user_data.user_query, user_data.history)  
        response = {"analysis_results": analysis_results}  
        print(response)
        return response  
    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))  
