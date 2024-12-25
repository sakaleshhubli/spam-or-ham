Spam or Ham: AI Message Classifier

Spam or Ham is a sleek, AI-powered tool that classifies messages as **Spam** or **Ham** (Not Spam) with ease and accuracy. Built using Logistic Regression and integrated with an interactive Streamlit web app, it allows you to upload datasets, visualize data trends, and make real-time predictions. This project is perfect for understanding spam detection and exploring machine learning in action.

## Features

### 📊 Analyze Your Data
- Upload your dataset in CSV format.
- The app provides detailed dataset insights, such as row/column counts, missing values, and data distributions.
- Visualize your dataset with insightful graphs, including bar charts and word clouds.

### 🤖 Smart Predictions
- Type any message and instantly predict whether it’s spam or ham.
- Powered by a Logistic Regression model, trained to deliver accurate predictions.

### ✨ Interactive Visuals
The app includes multiple visualizations to help you better understand your dataset and model performance:
1. **Spam vs. Ham Distribution**: A bar chart showing the ratio of spam to ham messages.
2. **Word Clouds**: Displays the most common words in spam and ham messages.
3. **ROC Curve**: Visualizes the model's ability to distinguish between spam and ham.
4. **Confusion Matrix**: Highlights true positives, true negatives, false positives, and false negatives.


## How It Works

### Step 1: Upload a Dataset
- Upload a CSV file containing your message data.
- The dataset should have two columns: one for the labels (spam/ham) and one for the messages.

### Step 2: Data Cleaning
- The app automatically cleans the data by removing unnecessary columns and handling missing values.
- Labels are converted into numerical values: 0 for ham and 1 for spam.

### Step 3: Feature Extraction
- Messages are vectorized using `HashingVectorizer` to convert text into numerical features suitable for machine learning.

### Step 4: Model Training
- The app splits the dataset into training and testing sets.
- Features are standardized using `StandardScaler`.
- A Logistic Regression model is trained on the processed data.

### Step 5: Evaluate and Visualize
- The app calculates the model’s accuracy and displays key metrics.
- Visualizations help you understand the model’s performance and the dataset’s characteristics.

### Step 6: Predict Custom Messages
- Enter a custom message to classify it as spam or ham instantly.

---

## Installation and Setup

Follow these steps to run the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/spam-or-ham.git
   cd spam-or-ham
   ```

2. **Install Dependencies**:
   Make sure you have Python 3.7 or higher installed. Then, install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. **Upload Your Dataset**:
   - Use the app interface to upload your CSV file.
   - Explore your data, visualize it, and predict spam messages.

---

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve this project.
