import pandas as pd
import os


def create_embeddings(db_name: str) -> list[str]:
    embeddings = []
    path = os.path.join("databases", db_name)
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
            
            embeddings.append(f"{name}, {new_row[:len(new_row)-2]}")

    return embeddings

def format_embeddings(embeddings: list[str]) -> dict:
    result = {}

    rows = []
    for row in embeddings:
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