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
        "What are the benefits of a balanced diet?", "How does machine learning work?",
        "Tips for improving sleep quality?", "What are the effects of climate change?",
        "How to start investing in stocks?", "Explain Newton’s laws of motion.",
        "What are the advantages of electric cars?", "How to improve time management skills?",
        "What are some easy yoga poses for beginners?", "Tell me about the history of Ancient Greece."
    ],
    "url_to_check": [
        "https://www.nasa.gov/mars-missions", "https://www.foodnetwork.com/best-desserts",
        "https://www.worldbank.org/economic-growth", "https://www.fifa.com/world-cup-history",
        "https://www.nobelprize.org/physics", "https://www.who.int/mental-health",
        "https://www.medicalnewstoday.com/heart-health", "https://www.techcrunch.com/startups",
        "https://www.bbc.com/history-world-war", "https://www.cnn.com/latest-tech"
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
