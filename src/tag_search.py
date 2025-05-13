import os
import json
from llm import query_groq
from embedder import Embedder
from milvusDB import MilvusDB
from data_manipulation import entry_to_dict, prepare_table_prompt
import paths

def get_llm_response(query: str, tables: str) -> str:

    """
    Function to extract relevant tables and attributes from natural language query.
    
    Args:
        query (str): Natural language query.
        tables (str): tables in the format preferred by the llm
    Returns:
        str: relevant and augmented data for user.
    """

    system_prompt = """You are a highly capable language model that answers natural language questions using only the data provided in tabular format.

You will be given:
1. A natural language question (query).
2. One or more tables in plain text. Each table starts with 'TABLE' followed by its name, then a header row with column names (comma-separated), and then multiple rows of data.

Your task is to:
- Analyze the user's question carefully.
- Use only the data in the provided tables to answer.
- Identify relevant tables and columns.
- Apply operations such as filtering, counting, sorting, or aggregating as required to compute the correct answer.
- If answering the question requires combining data from multiple tables, perform a join operation using shared identifiers or keys (e.g., matching rows by 'id').
- Do not use any external knowledge or assumptions.
- If the answer cannot be derived from the available data, respond with: "The answer cannot be determined from the provided data."
- Keep your answer short, clear, and focused on what the user asked.

Example format:

TABLE 'employees'
id,name,department
1,Alice,Engineering
2,Bob,Marketing
3,Charlie,Engineering

TABLE 'salaries'
id,salary
1,70000
2,65000
3,72000

You must strictly base your reasoning only on the tables provided."""

    content = f"""QUESTION:
{query}

TABLES:
{tables}
"""


    return query_groq(messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ])


if __name__ == '__main__':

    embedder = Embedder()
    os.makedirs(paths.LLM_RESPONSE, exist_ok=True)
    gt = os.listdir(paths.GROUND_TRUTH)
    for db in gt:
        file_path = os.path.join(paths.GROUND_TRUTH, db)
        db_name, ext = os.path.splitext(db)

        milvusdb = MilvusDB(embedder, db_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(paths.LLM_RESPONSE + db_name + '.txt', "a", encoding="utf-8") as file:
            for item in data:
                q_id = item['question_id']
                db_id = item['db_id']
                query = item['question']
                sql = item['SQL']

        #========== Embeddings similarity phase ==========

                results = milvusdb.search(query, threshold=0.3)
                res = []

                for result in results:
                    res.append(result['text'])

                dict_res = entry_to_dict(res)
                prompt_tables = prepare_table_prompt(dict_res)

        #========== LLM generation phase =================

                llmResponse = get_llm_response(query, prompt_tables)

        #=================================================

                file.write(f"Qid: {q_id}\n")
                file.write(f"DBid: {db_id}\n")
                file.write(f"QUESTION: {query}\n")
                file.write(f"SQL: {sql}\n")
                file.write(f"TP: \n")
                file.write(f"FP: \n")
                file.write(f"FN: \n")
                file.write(f"uScore(1-5): \n")
                file.flush()