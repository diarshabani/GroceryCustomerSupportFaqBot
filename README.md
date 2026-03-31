# Grocery Store FAQ Chatbot

Grocery Inventory FAQ chatbot that leverages a CSV dat set, python flask api and Open AI apis to help assist customers

## Prerequisites

- Python 3.8+
- OpenAI API key available as the `OPENAI_API_KEY` environment variable

## Installation

Install the required packages:

```bash
pip install langchain langchain-openai langchain-community faiss-cpu flask pandas openai
```

## Usage

1. Build or refresh the FAISS index:

   ```bash
   python3 data_pipeline.py
   ```

2. Start the Flask application:

   ```bash
   python3 app.py
   ```

3. Open the web interface in your browser:

   ```text
   http://localhost:5000
   ```

4. Run the performance benchmark script:

   ```bash
   python3 test_performance.py
   ```

## Project Structure

Project root: `/Users/diorock/Projects/CS311_AI_StudyDotCom/Assignment2`

```text
/Users/diorock/Projects/CS311_AI_StudyDotCom/Assignment2
├── .venv/                        # Optional local virtual environment
├── Grocery_Inventory_and_Sales_Dataset.csv
├── README.md
├── app.py
├── chatbot.py
├── data_pipeline.py
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── templates/
│   └── index.html
└── test_performance.py
```

## Dataset Source

The chatbot usees a local dataset file imported from kaggle.com, Grocery_Inventory_and_Sales_Dataset.csv. The dataset shows a snapshot of what an inventory management system may look like in a real world setting. Kaggle source : https://www.kaggle.com/datasets/salahuddinahmedshuvo/grocery-inventory-and-sales-dataset

## Runtime Notes

- data_pipeline.py, app.py and test_performance.py require a valid OPENAI_API_KEY on your local environment configured.
- We leverage FAISS index for the chatbot to quickly query / acquire inventory information to then convey to the user.
- If FAISS index is missing or the OpenAI api request is not able to be completed, chatbot.py will fallback to answering common product, category price and status questions directly from the CSV leveraging a keyword fuzzy match to try to address common user queries backed by data that exists in the CSV
- the webapp loads at localhost:5000 once running
