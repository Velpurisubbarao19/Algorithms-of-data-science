import os
import pandas as pd
import random

def calculate_score(content_relevance: int, bias_score: int) -> int:
    """
    Calculates a simple final score based on content relevance and bias score.

    Input args:
      content_relevance: int - Score indicating how relevant the content is to the query.
      bias_score: int - Score indicating the neutrality or bias of the content.

    Return:
      final_score: int - A weighted average of the two scores.
    """
    return int((0.5 * content_relevance) + (0.5 * bias_score))

# Dataset
data = {
    "user_prompt": [
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
    ],
    "url_to_check": [
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
}

# Generating scores with random values for simulation
data["func_rating"] = [calculate_score(random.randint(0, 100), random.randint(0, 100)) for _ in range(len(data["user_prompt"]))]
data["custom_rating"] = [random.randint(1, 5) for _ in range(len(data["user_prompt"]))]

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to desktop
desktop_path = os.path.expanduser("~/Desktop")
os.makedirs(desktop_path, exist_ok=True)
csv_file_path = os.path.join(desktop_path, "deliverable.csv")
df.to_csv(csv_file_path, index=False)

print(f"Deliverable CSV file saved at: {csv_file_path}")
