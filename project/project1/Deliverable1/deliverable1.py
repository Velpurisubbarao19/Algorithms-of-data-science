import os
import pandas as pd
import requests
import random
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

def evaluate_url_quality(query_text: str, webpage: str) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(webpage, headers=headers, timeout=10)
        response.raise_for_status()
        parsed_html = BeautifulSoup(response.text, "html.parser")
        extracted_text = " ".join([p.text for p in parsed_html.find_all("p")])
    except Exception as err:
        print(f"Error fetching webpage {webpage}: {err}")
        return {  
            "Text Relevance": 0, 
            "Neutrality Score": 0, 
            "Overall Quality Score": 0
        }

    if not extracted_text.strip():
        print(f"Warning: No content extracted from {webpage}")
        return {  
            "Text Relevance": 0, 
            "Neutrality Score": 0, 
            "Overall Quality Score": 0
        }

    encoder_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    relevance = util.pytorch_cos_sim(encoder_model.encode(query_text), encoder_model.encode(extracted_text)).item() * 100

    sentiment_analyzer = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
    sentiment_outcome = sentiment_analyzer(extracted_text[:512])[0]
    neutrality = 100 if sentiment_outcome["label"] == "POSITIVE" else 50 if sentiment_outcome["label"] == "NEUTRAL" else 30

    overall_score = (0.5 * relevance) + (0.5 * neutrality)

    print(f"Debug: Computed Quality Score for {webpage} -> {overall_score}")

    return {  
        "Text Relevance": relevance,
        "Neutrality Score": neutrality,
        "Overall Quality Score": overall_score
    }

# Directory setup
storage_path = os.path.expanduser("~/Desktop")
if not os.path.exists(storage_path):
    os.makedirs(storage_path)

csv_output_path = os.path.join(storage_path, "final_results.csv")

query_samples = [
    "What are the benefits of yoga?",
    "How to cook a perfect steak?",
    "History of the Renaissance period.",
    "Side effects of aspirin.",
    "Effective home workouts.",
    "How to boost memory power?",
    "Recent breakthroughs in AI?",
    "Best techniques for job interviews.",
    "Basics of quantum mechanics.",
    "Recommended books for beginners in programming.",
    "Tips for budgeting and saving money.",
    "Health benefits of mindfulness meditation.",
    "Symptoms of seasonal allergies.",
    "How does cryptocurrency work?",
    "Understanding black holes."
]

web_links = [
    "https://www.cnn.com/world",
    "https://www.bbc.com/news/technology",
    "https://www.scientificamerican.com",
    "https://www.who.int/news-room",
    "https://www.cdc.gov/health/index.html",
    "https://www.nasa.gov/news",
    "https://en.wikipedia.org/wiki/History_of_science",
    "https://www.java.com/en/",
    "https://www.tesla.com/AI",
    "https://www.nature.com/latest-news",
    "https://www.mayoclinic.org/health-info",
    "https://www.sciencenews.org",
    "https://www.nationalgeographic.com/science",
    "https://www.forbes.com/innovation",
    "https://www.wsj.com/technology"
]

num_entries = 10
data_collection = {
    "query_statement": random.sample(query_samples, num_entries),
    "web_address": random.sample(web_links, num_entries),
    "computed_score": [evaluate_url_quality(query, link).get("Overall Quality Score", 0) for query, link in zip(query_samples[:num_entries], web_links[:num_entries])],
    "manual_rating": [random.randint(1, 5) for _ in range(num_entries)]
}

dataframe = pd.DataFrame(data_collection)

# Append to existing CSV if available
existing_csv_path = os.path.join(storage_path, "previous_results.csv")
try:
    existing_df = pd.read_csv(existing_csv_path)
    combined_df = pd.concat([existing_df, dataframe], ignore_index=True)
except FileNotFoundError:
    combined_df = dataframe

combined_df.to_csv(csv_output_path, index=False)
print(f"Final results CSV saved at: {csv_output_path}")

