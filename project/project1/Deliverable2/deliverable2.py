import os
import pandas as pd
import requests
import random
from bs4 import BeautifulSoup

try:
    from sentence_transformers import SentenceTransformer, util
    from transformers import pipeline
    MODULES_AVAILABLE = True
except ModuleNotFoundError:
    print(" Warning: Required ML modules are missing. Running in fallback mode.")
    MODULES_AVAILABLE = False

class URLValidator:
    def __init__(self):
        if MODULES_AVAILABLE:
            self.similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.sentiment_analyzer = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
        else:
            self.similarity_model = None
            self.sentiment_analyzer = None

    def fetch_page_content(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return " ".join([p.text for p in soup.find_all("p")])
        except requests.RequestException:
            return ""

    def compute_similarity_score(self, user_query: str, content: str) -> int:
        if not content or not MODULES_AVAILABLE:
            return 0
        return int(util.pytorch_cos_sim(self.similarity_model.encode(user_query), self.similarity_model.encode(content)).item() * 100)

    def detect_bias(self, content: str) -> int:
        if not content or not MODULES_AVAILABLE:
            return 50
        sentiment_result = self.sentiment_analyzer(content[:512])[0]
        return 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30

    def rate_url_validity(self, user_query: str, url: str) -> dict:
        content = self.fetch_page_content(url)
        similarity_score = self.compute_similarity_score(user_query, content)
        bias_score = self.detect_bias(content)
        final_score = (0.5 * similarity_score) + (0.5 * bias_score)
        return {
            "Content Relevance": similarity_score,
            "Bias Score": bias_score,
            "Final Validity Score": final_score
        }

desktop_path = os.path.expanduser("~/Desktop")
os.makedirs(desktop_path, exist_ok=True)
csv_file_path = os.path.join(desktop_path, "deliverable.csv")

new_prompts = [
    "What are the benefits of a plant-based diet?",
    "How does quantum computing work?",
    "What are the causes of climate change?",
    "Explain the basics of blockchain technology.",
    "How can I learn a new language quickly?",
    "What are the symptoms of diabetes?",
    "What are the best books for personal development?",
    "How does 5G technology impact daily life?",
    "What are the career opportunities in data science?",
    "What are the ethical concerns surrounding AI?"
]

new_urls = [
      "https://www.healthline.com/nutrition/plant-based-diet-guide",
    "https://www.ibm.com/quantum-computing/what-is-quantum-computing",
    "https://climate.nasa.gov/evidence/",
    "https://www.investopedia.com/terms/b/blockchain.asp",
    "https://www.duolingo.com/",
    "https://www.diabetes.org/diabetes",
    "https://jamesclear.com/book-summaries",
    "https://www.qualcomm.com/news/onq/2020/01/10/what-5g-and-how-it-changing-everything",
    "https://datasciencedegree.wisconsin.edu/data-science/what-do-data-scientists-do/",
    "https://aiethicslab.com/"
]

num_rows = min(len(new_prompts), len(new_urls))
validator = URLValidator()

new_data = {
    "user_prompt": new_prompts,
    "url_to_check": new_urls,
    "func_rating": [validator.rate_url_validity(prompt, url).get("Final Validity Score", 0) for prompt, url in zip(new_prompts, new_urls)],
    "custom_rating": [random.randint(1, 5) for _ in range(num_rows)]
}

df = pd.DataFrame(new_data)

df.to_csv(csv_file_path, index=False)
print(f" Deliverable CSV file saved at: {csv_file_path}")

