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
        length = len(columns)
        
        for row in df.itertuples():
            embedding_value = [value for value in row[2:]]
            embedding_column = [column for column in columns[1:]]

            new_row = ""
            for i, (column, value) in enumerate(zip(embedding_column, embedding_value)):
                new_row += f"{column}: {value}, "
            
            embeddings.append(f"{name}, {new_row[:len(new_row)-2]}")

    return embeddings