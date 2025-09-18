# Import required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Small dataset of emails
emails = [
    "Get rich Quick! Click here to win a million dollars!",
    "Hello, could you please review this document for me",
    "Discounts on luxury watches and handbags!",
    "Meeting scheduled for tomorrow, please confirm your attendance.",
    "Congratulations, you've won a free gift card!"
]

labels = [1, 0, 1, 0, 1]  # 1 = spam, 0 = not spam

# Convert text data into numerical features using Count Vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create a Multinomial Naive Bayes classifier
model = MultinomialNB()

# Train the model on training data
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Predict whether a new email is spam or not
new_email = ["You've won a free cruise vacation"]
new_email_vectorized = vectorizer.transform(new_email)
predicted_label = model.predict(new_email_vectorized)

if predicted_label[0] == 0:
    print("Predicted as NOT spam.")
else:
    print("Predicted as SPAM.")


