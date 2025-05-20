# TAG-easy-relational-access
TAG (Table Augmented Generation) is an application that, upon receiving natural language data analysis requests, constructs a prompt for a Large Language Model (LLM) using the content of data stored in a relational database.

This experiment was conducted on a subset of databases from the BIRD (dev) dataset and aims to introduce an additional layer of abstraction between the user and SQL, enabling natural language querying.

## Replicate the experiment

Install Python 3.11.11. Execute the following command.

```bash
git clone https://github.com/aledigirm3/TAG-easy-relational-access.git
cd TAG-easy-relational-access
pip install -r requirements.txt
```

Before running the code, make sure Docker is installed and running. Then, launch the necessary services: 

```bash
  docker compose up -d
```

After confirming that the MilvusDB containers are up and running correctly, you may proceed with the core experiment.

Navigate to the 'src' directory:

```bash
  cd src
```

 Before executing the scripts, you must create a .env file in the root directory of the project. Use the structure provided in the .env.example file, replacing 'API_KEY' with your personal key obtained from Groq.
```env
# Example .env file
API_KEY=your_groq_api_key_here
```
#### ⚠️ Important:
To successfully run the experiment, you must have access to Groq's Developer Tier, which supports pay-per-token usage. Lower tiers or trial access may not be sufficient.

Now run these scripts (in order as shown)

```bash
  python data_manipulation.py
  python tag_search.py
```

The first script performs the following operations (for each database individually):

- Retrieves all rows from all tables, processing them into a structured format
- Computes embeddings for each processed row
- Saves the embeddings into a vector database. A separate collection is created in MilvusDB for each database

The second script performs:

- Retrieval of entries from the 'gt' folder
- For each entry, it searches for the most semantically similar table rows (using embeddings)
- Passes the entry's query and the retrieved table rows to the LLM, which generates a refined and reasoned response

## Evaluation

The evaluation of the system is conducted manually. As such, the responses saved include fields for various performance metrics, including the uScore, which represents a general human-assigned quality score of the LLM’s response.