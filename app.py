import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from wordcloud import WordCloud

# Title of the web app
st.title("Spam Classification with Logistic Regression")

# Step 1: Upload the dataset
st.write("### Step 1: Upload your dataset (CSV format)")

uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file, encoding="latin1")
    
    # Show the dataset's basic information and first few rows
    st.write("### Dataset Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write("### Dataset Preview")
    st.write(df.head())

    # Data cleaning and processing
    if "Unnamed: 2" in df.columns:
        df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
    
    df.columns = ["spam or not", "message"]
    df["spam or not"] = df["spam or not"].replace({"ham": 0, "spam": 1})
    df = df.dropna()

    # Display cleaned data preview
    st.write("### Cleaned Dataset Preview")
    st.write(df.head())
    
    # Step 2: Vectorize the text data using HashingVectorizer
    vectorizer = HashingVectorizer(n_features=1024)
    hashed_features = vectorizer.transform(df["message"])
    hashed_features_dense = hashed_features.toarray()

    # Step 3: Split data into training and testing sets
    X = hashed_features_dense
    y = df["spam or not"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 4: Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Train the Logistic Regression model
    clf = LogisticRegression(random_state=0, max_iter=200)
    clf.fit(X_train_scaled, y_train)

    # Step 6: Predict and calculate accuracy
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"### Model Accuracy: {accuracy:.2f}")

    # Step 7: Visualizations
    # 1. Distribution of Spam and Not Spam Messages (Bar Chart)
    st.write("### 1. Distribution of Spam and Not Spam Messages")
    plt.figure(figsize=(6, 4))
    df["spam or not"].value_counts().plot(kind="bar", color=["green", "red"])
    plt.title("Spam vs Not Spam")
    plt.xticks([0, 1], ['Not Spam', 'Spam'], rotation=0)
    plt.ylabel("Count")
    st.pyplot(plt)
    st.write("This bar chart shows the distribution of spam vs. non-spam messages in the dataset.")

    # 2. Word Cloud of Most Frequent Words in Spam and Non-Spam Messages
    st.write("### 2. Word Cloud of Most Frequent Words in Spam and Non-Spam Messages")
    spam_messages = df[df["spam or not"] == 1]["message"].str.cat(sep=" ")
    non_spam_messages = df[df["spam or not"] == 0]["message"].str.cat(sep=" ")

    wordcloud_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_messages)
    wordcloud_non_spam = WordCloud(width=800, height=400, background_color='white').generate(non_spam_messages)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(wordcloud_spam, interpolation='bilinear')
    axes[0].set_title("Word Cloud for Spam Messages")
    axes[0].axis('off')

    axes[1].imshow(wordcloud_non_spam, interpolation='bilinear')
    axes[1].set_title("Word Cloud for Non-Spam Messages")
    axes[1].axis('off')

    st.pyplot(fig)
    st.write("These word clouds show the most frequent words in spam and non-spam messages. Spam messages tend to use more sales and prize-related terms.")

    # 3. ROC Curve
    st.write("### 3. ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    st.write("The ROC curve helps evaluate the performance of the model. A higher AUC (Area Under the Curve) means better performance.")

    # 4. Confusion Matrix
    st.write("### 4. Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center', color='red', fontsize=16)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks([0, 1], ['Not Spam', 'Spam'], rotation=0)
    plt.yticks([0, 1], ['Not Spam', 'Spam'], rotation=0)
    st.pyplot(fig)
    st.write("This confusion matrix shows the counts of true positive, true negative, false positive, and false negative predictions.")

    # Step 8: Predict a custom message
    st.write("### Step 2: Predict if a custom message is spam or not")
    user_input = st.text_area("Enter a message to classify:", "")
    if user_input:
        user_vectorized = vectorizer.transform([user_input])
        user_vectorized_dense = user_vectorized.toarray()
        user_scaled = scaler.transform(user_vectorized_dense)
        prediction = clf.predict(user_scaled)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.write(f"The message is: {result}")
