import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Sample Data
data = {
    'employee_id': [1, 2, 3, 4, 5],
    'job_role': ['Engineer', 'Manager', 'Analyst', 'Developer', 'Designer'],
    'temperature': [22, 21, 23, 22, 21],
    'light_levels': [300, 320, 280, 310, 305],
    'air_quality': [40, 35, 45, 50, 42],
    'feedback': [
        "I feel great today!",
        "The room is too cold.",
        "Air quality is good but lighting is too dim.",
        "Perfect working conditions.",
        "The lighting could be better."
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Sentiment Analysis
df['sentiment'] = df['feedback'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Adjust environmental controls based on sentiment
def adjust_environment(sentiment):
    if sentiment > 0.5:
        return "Increase Temperature"
    elif sentiment < -0.5:
        return "Decrease Temperature"
    else:
        return "Maintain Temperature"

df['adjustment'] = df['sentiment'].apply(adjust_environment)
print(df[['employee_id', 'sentiment', 'adjustment']])

# Plot Sentiment
plt.bar(df['employee_id'], df['sentiment'], color='blue')
plt.xlabel('Employee ID')
plt.ylabel('Sentiment')
plt.title('Employee Sentiment Analysis')
plt.show()
