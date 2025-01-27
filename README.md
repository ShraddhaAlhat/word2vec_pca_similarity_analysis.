# Word2Vec + PCA for Word Similarity

## Project Overview
This project demonstrates the process of training a custom Word2Vec model, applying Principal Component Analysis (PCA) for dimensionality reduction, visualizing the word embeddings in 2D, and calculating word similarity. The steps covered in this project are as follows:

1. **Dataset Collection**: The dataset used for training the Word2Vec model is sourced from Kaggle.
2. **Training Word2Vec Model**: The custom Word2Vec model is trained on the collected dataset to generate word embeddings.
3. **Dimensionality Reduction (PCA)**: PCA is applied to reduce the high-dimensional word vectors to 2 dimensions for easier visualization.
4. **Data Visualization**: The 2D reduced data is visualized using a scatter plot to see the relationships and clustering of words.
5. **Word Similarity**: Cosine similarity is used to measure the similarity between words in the Word2Vec and PCA-transformed spaces.

## Dataset
- **Source**: [Harry Potter Books](https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books)
- **Description**: This data consists of textual data to train the Word2Vec model.

## Steps Followed
### 1. Data Preprocessing
- Cleaned and tokenized the text data.
- Removed stop words, non-alphabetical characters, and unnecessary punctuation.

### 2. Training the Word2Vec Model
- Used `gensim`'s Word2Vec implementation to train the model on the text data.
- The model was trained to produce word vectors based on the context of each word.

### 3. Dimensionality Reduction using PCA
- Reduced the dimensionality of the word vectors from 100 (or more) dimensions to 2 using PCA for visualization purposes.
- PCA helps in visualizing high-dimensional data in a 2D space while retaining most of the variance.

### 4. Visualization of Word Vectors
- The 2D vectors obtained from PCA are plotted in a scatter plot, allowing the visualization of relationships between words.

### 5. Word Similarity Calculation
- Used cosine similarity to measure how similar two words are in the Word2Vec space and the PCA-reduced space.
- Words with similar meanings are expected to be closer together in the vector space.

## Libraries Used
- `gensim`: For training the Word2Vec model.
- `sklearn`: For PCA implementation.
- `matplotlib`: For data visualization.

## Results
- **Word2Vec**: Cosine similarity between words shows how semantically related they are in the trained Word2Vec space.
- **Word2Vec + PCA**: By applying PCA, the relationships between words are visualized in 2D, and cosine similarity can be computed for words in this reduced space as well.
- ## PCA Visualization
<p align="center">
  <img src="https://github.com/ShraddhaAlhat/word2vec_pca_similarity_analysis./commit/17ff29e1b06aaddce626cac1cd9df71adfb66444" alt="Image Description" width="300" height="200">
</p>
## Conclusion
This project provides a complete workflow for training word embeddings using Word2Vec, reducing the dimensionality using PCA, visualizing the word vectors, and calculating word similarities. These techniques are foundational for many NLP applications like document clustering, recommendation systems, and information retrieval.


