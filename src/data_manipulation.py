import pandas as pd
import os
from milvusDB import MilvusDB
import json


def create_embeddings():
    """
    Extracts all rows from every table in every database, generates embeddings for the row data,
    and stores the resulting embeddings in a vector database (Milvus).
    """

    milvusdb = MilvusDB()
    pre_embeddings_rows = []

    databases = sorted([
    d for d in os.listdir('../databases')
    if os.path.isdir(os.path.join('../databases', d))
])
    for db_name in databases:
        path = os.path.join("../databases", db_name)
        tables_names = sorted(os.listdir(path))
        
        for table_name in tables_names:
            name = table_name.split(".")[0]
            df = pd.read_csv(os.path.join(path, table_name))
            columns = df.columns.to_list()
            
            for row in df.itertuples():
                embedding_value = [value for value in row[2:]]
                embedding_column = [column for column in columns[1:]]

                new_row = ""
                for i, (column, value) in enumerate(zip(embedding_column, embedding_value)):
                    new_row += f"{column}: {value}, "
                
                pre_embeddings_rows.append(f"{name}, {new_row[:len(new_row)-2]}")

    milvusdb.add_texts(pre_embeddings_rows)

def format_embeddings(pre_embeddings_rows: list[str]) -> dict:
    """
    Function that read the `pre_embeddings_rows` and return a dict that organize each db
    
    Args:
        pre_embeddings_rows (list): list where each element is formatted like this: `db_name`, `column1: value1`, `column2: value2`, .... 
        
    Returns:
        dict: dictionary in this format:
            {
                "db_name": {
                    "attributes": "",
                    "rows" : []
                }
            }
    """
    result = {}

    rows = []
    for row in pre_embeddings_rows:
        row = row.split(", ")
        db_name = row.pop(0)
        row = ",".join(row).split(",")

        attributes = []
        values = []

        for element in row:
            index = element.find(": ")
            attribute = element[:index]
            value = element[index+2:]
            
            attributes.append(attribute)
            values.append(value)

        rows.append((db_name, attributes, values))
    
    for (db_name, attributes, values) in rows:
        if db_name not in result:
            db = dict(
                attributes = ",".join(attributes),
                rows = [",".join(values)]
            )
            result[db_name] = db
        else:
            db = result[db_name]
            db["rows"].append(",".join(values))

            result[db_name] = db 

    return result

def get_simple_query(json_path: str):
    dev = json.load(open(json_path))

    result = {}

    for question in dev:
        question_id = question["question_id"]
        SQL:str = question["SQL"]
        db_id = question["db_id"]

        if "JOIN" not in SQL:
            if db_id in result:
                result[db_id].append(question_id)
            else:
                result[db_id] = [question_id]

    return result

if __name__ == '__main__':

    milvusdb = MilvusDB()

    query = "What is Copycat's race?"

    with open("out.txt", "w") as file:
        results = milvusdb.search(query, threshold=0.2)
        for result in results:
            file.write(f"{json.dumps(result)}\n")
        print(len(results))

    #print(get_simple_query("../databases/dev.json"))