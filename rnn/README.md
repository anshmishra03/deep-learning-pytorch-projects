# IMDB Sentiment Analysis using RNN

Binary sentiment classifier that predicts whether an IMDB movie review is
positive or negative, using a Recurrent Neural Network built in PyTorch.

## Dataset
IMDB Movie Reviews dataset (50,000 reviews, labelled positive / negative).
After dropping duplicate reviews, 49,582 reviews were used.

## Approach

**1. Data cleaning**
- Checked for missing values (none found)
- Removed duplicate reviews

**2. Text preprocessing**
- Converted text to lowercase
- Removed URLs, HTML tags and punctuation (using regex)
- Removed English stopwords (NLTK)
- Applied stemming (NLTK PorterStemmer)

**3. Feature preparation**
- Encoded sentiments with LabelEncoder (negative = 0, positive = 1)
- Vectorized text using TF-IDF (top 5,000 features)
- 80/20 train-test split (39,665 train / 9,917 test)
- Converted data to PyTorch tensors, built TensorDatasets and DataLoaders (batch size 64)

**4. Model**
- RNN built with `torch.nn.RNN` (hidden size 128, 1 layer) followed by a fully connected output layer
- Sigmoid output with Binary Cross-Entropy loss
- Adam optimizer, trained for 10 epochs

**5. Evaluation**
- Evaluated on the held-out test set using accuracy

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 82.44% |

## Tech Stack
Python, PyTorch, NLTK, Scikit-learn, Pandas, NumPy, Jupyter
