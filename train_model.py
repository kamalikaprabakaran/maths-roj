import pandas as pd
import joblib
import random
import re
import nltk
import csv
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Ensure stopwords are available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Define distributions and question templates
distributions = {
    "binomial": [
        "A coin is flipped {} times. What is the probability of getting exactly {} heads?",
        "A factory tests {} items with a {}% defect rate. What is the probability of exactly {} defective items?"
    ],
    "poisson": [
        "A shop gets {} customers per hour. What is the probability that {} customers arrive in an hour?",
        "A call center receives {} calls per minute. What is the probability of exactly {} calls?"
    ],
    "geometric": [
        "A player has a {}% success rate. What is the probability that the first success occurs on the {}th attempt?",
        "A factory machine assembles an item successfully {}% of the time. What is the probability that the first success happens on the {}th attempt?"
    ],
    "exponential": [
        "A light bulb lasts an average of {} hours. What is the probability that it lasts more than {} hours?",
        "A bus arrives every {} minutes on average. What is the probability that it arrives within {} minutes?"
    ],
    "normal": [
        "Heights of students follow a normal distribution with mean {} cm and SD {} cm. What is the probability of a student being shorter than {} cm?",
        "A test has an average score of {} with a standard deviation of {}. What is the probability of scoring above {}?"
    ],
    "uniform": [
        "A number is randomly picked between {} and {}. What is the probability that it is less than {}?",
        "A bus randomly arrives between {} and {} minutes. What is the probability of arriving before {} minutes?"
    ]
}

# Generate dataset with 200 questions covering all distributions
dataset = []
num_questions_per_distribution = 200 // len(distributions)

for dist, templates in distributions.items():
    for _ in range(num_questions_per_distribution):
        template = random.choice(templates)
        values = [random.randint(5, 100) for _ in range(template.count("{}"))]
        problem = template.format(*values)
        dataset.append([problem, dist])

# Save dataset to CSV
dataset_file = "dataset.csv"
with open(dataset_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["problem_statement", "distribution"])
    writer.writerows(dataset)

print(f"✅ Dataset saved with {len(dataset)} questions.")

# Load dataset for preprocessing
df = pd.read_csv(dataset_file, encoding="latin1")

# Text Preprocessing: Lowercase, remove special characters, remove stopwords
df["problem_statement"] = df["problem_statement"].str.lower()
df["problem_statement"] = df["problem_statement"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))
df["problem_statement"] = df["problem_statement"].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

# Save cleaned dataset
cleaned_dataset_file = "cleaned_dataset.csv"
df.to_csv(cleaned_dataset_file, index=False)
print(f"✅ Data preprocessing completed. Cleaned dataset saved as '{cleaned_dataset_file}'.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["problem_statement"], df["distribution"], test_size=0.2, random_state=42)

# NLP pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced"))
])

# Train model
pipeline.fit(X_train, y_train)

# Save trained model
model_file = "distribution_model.pkl"
joblib.dump(pipeline, model_file)

# Evaluate model
accuracy = pipeline.score(X_test, y_test)
print(f"✅ Model training complete. Accuracy: {accuracy * 100:.2f}%")
print(f"📂 Trained model saved as '{model_file}'")

#douhfi