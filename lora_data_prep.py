import os
import json
import random
import pandas as pd
random.seed(42)

DATA_DIR = "data"

# Load the customer support data
df = pd.read_csv(os.path.join(DATA_DIR, "aa_dataset-tickets-multi-lang-5-2-50-version.csv"))

# Remove rows with missing values
df = df.dropna(subset=['subject', 'body', 'queue', 'type'])
df.head()

# Set your split ratios
TRAIN_RATIO = 0.9
VAL_RATIO = 0.09
TEST_RATIO = 0.01

PREPARED_DATA_DIR = os.path.join(DATA_DIR, "prepared-data")
os.makedirs(PREPARED_DATA_DIR, exist_ok=True)


# This list will hold all of our transformed data points.
transformed_data = []

def create_prompt(subject, body):
    """
    Creates a standardized prompt for the language model.
    """
    return f"A customer has submitted a support ticket. Please route it to the correct department.\n\nSubject: {subject}\n\nBody: {body}\n\nDepartment:"


# Iterate over each row of the DataFrame to create the prompt-completion pairs.
for index, row in df.iterrows():
    prompt = create_prompt(row['subject'], row['body'])
    # completion = row['type'] + ", " + row['queue']
    completion = row['queue']
    
    transformed_data.append({
        "input": prompt,
        "output": f"{completion}"
    })


random.shuffle(transformed_data)
n = len(transformed_data)

# Calculate split indices
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_data = transformed_data[:train_end]
val_data = transformed_data[train_end:val_end]
test_data = transformed_data[val_end:]

# Determine folder


def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

# Save each split
save_jsonl(train_data, os.path.join(PREPARED_DATA_DIR, "training.jsonl"))
save_jsonl(val_data, os.path.join(PREPARED_DATA_DIR, "validation.jsonl"))
save_jsonl(test_data, os.path.join(PREPARED_DATA_DIR, "test.jsonl"))

print(f"Total records: {n}")
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
print(f"Saved to {PREPARED_DATA_DIR}")