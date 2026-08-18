# Grocery Store FAQ Chatbot assitant

Based off of Grocery Store CSV data, we created Grocery Inventory FAQ chatbot that is ran through python flask api and OpenAI APIs to help assist customers and staff members of the grocery store.


## Prerequisites

- Python 3.8+
- an OpenAI API key exported as the `OPENAI_API_KEY` environment variable:

  ```bash
  export OPENAI_API_KEY="your-key-here"
  ```

## Installation

Install the required python packages:

```bash
pip install langchain langchain-openai langchain-community faiss-cpu flask pandas openai
```

## Usage

1. Build or refresh the FAISS index based off the grocery store inventory data:

   ```bash
   python3 data_pipeline.py
   ```

2. Start the Flask application to serve the web ui and backend:

   ```bash
   python3 app.py
   ```

3. Open the web app on your browser by navigating to :

   ```text
   http://localhost:5000
   ```

4. additionall you can run a performance benchmark script:

   ```bash
   python3 test_performance.py
   ```

## Project Structure

All paths are relative to the project root (the `Assignment2` folder):

```text
Assignment2/
├── .venv/                        # Optional local virtual environment
├── Grocery_Inventory_and_Sales_Dataset.csv
├── README.md
├── app.py
├── chatbot.py
├── chatbot_assets.py
├── chatbot_config.json
├── data_pipeline.py
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── faq_assistant_prompt.txt
├── inventory_utils.py
├── templates/
│   └── index.html
└── test_performance.py
```

## Dataset Source

The chatbot uses a local csv imported from kaggle.com called Grocery_Inventory_and_Sales_Dataset.csv. This csv shows a large amount of data relating to the inventory of a Grocery store in a real world setting. Kaggle source : https://www.kaggle.com/datasets/salahuddinahmedshuvo/grocery-inventory-and-sales-dataset
