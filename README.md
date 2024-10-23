# LLM Recommender System

## Overview

This repository contains the code, data, and methodology used to build an advanced recommendation system leveraging Large Language Models (LLMs) and traditional collaborative filtering techniques. The project aims to predict how users would rate books they haven’t reviewed yet, enhancing user experience by offering personalized recommendations.

We incorporate **LLMs** to process and understand user reviews in natural language, generating insightful representations that can improve recommendation accuracy. The system adopts a hybrid approach:

- **Collaborative Filtering**: Recommends books based on user-item interaction patterns.
- **Content-Based Filtering with LLMs**: Uses text features from reviews processed through LLMs to capture semantic meanings and context in user feedback, thus enhancing recommendation accuracy.

This hybrid approach combines the strengths of collaborative filtering for personalization with the language understanding capabilities of LLMs, which enhances content-based filtering by improving the representation of reviews.

## Key Achievements

- Collaborative filtering model RMSE: **0.94**
- Content-based filtering model RMSE: **0.69** (significant improvement using LLM-processed features)
- LLM-powered review vectorization enables deeper understanding of user preferences.

The project demonstrates the potential of using advanced LLMs, like BERT, GPT-3, and their derivatives, for enhancing recommendation systems by generating semantic embeddings from reviews.

---

## Dataset

The dataset used for this project is sourced from Kaggle and contains detailed **Amazon book reviews**. This dataset is specifically tailored to the book category, making it an ideal candidate for building a recommendation system in the domain of literature.

### Dataset Details

- **Total Rows**: 3,105,370 (subsampled for this project)
- **Number of Features**: 15
- **Domain**: Books
- **Columns**:
  - `marketplace`: Marketplace where the review was written.
  - `customer_id`: Unique identifier for each customer.
  - `product_id`: Unique identifier for each book.
  - `product_title`: Title of the book.
  - `star_rating`: The rating given by the user (1-5 stars).
  - `review_headline`: Brief headline of the review.
  - `review_body`: Full text of the review.
  - `helpful_votes` and `total_votes`: User feedback on the helpfulness of the review.
  - `vine`: Whether the review was part of the Vine program.

The dataset is rich in textual content, making it an excellent fit for natural language processing (NLP) tasks like summarization, sentiment analysis, and semantic understanding. These features are extracted, summarized, and vectorized using LLMs to improve the recommendation system.

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/beaglelee/amazon-reviews-us-books-v1-02-tsv-zip).

---

## LLM Integration

In this project, LLMs play a critical role in the **content-based filtering** part of the recommendation system. We use pre-trained models like **BERT** (Bidirectional Encoder Representations from Transformers) to process and understand user reviews. The LLMs are utilized to:

1. **Summarize Reviews**: LLMs summarize long and detailed reviews into more concise representations, which helps in feature extraction.
2. **Create Semantic Vectors**: LLMs are used to generate dense vector embeddings for each review. These embeddings capture not just keywords but the semantic meaning of the text.
3. **Improve Recommendation Quality**: By incorporating LLM-generated review embeddings, the content-based filtering model can make more accurate predictions about books that are similar in both content and sentiment.

### Steps in LLM Integration:
- **Text Preprocessing**: Reviews are cleaned, tokenized, and prepared for input to the LLM.
- **Embedding Generation**: We pass each review through the LLM to generate an embedding that represents the review in vector space.
- **Review Summarization**: Long reviews are summarized for more efficient processing, while preserving important details.
- **Similarity Computation**: The embeddings are then used to compute similarities between different books, allowing the system to recommend books that are semantically similar based on their reviews.

By incorporating LLMs, we enhance the recommendation engine’s ability to understand user sentiment and book content, leading to more meaningful and accurate recommendations.

---

## Project Structure

The repository is organized as follows:

```
├── data/                             # Contains data files for the project
│   ├── amazon_reviews_subset.csv      # Subset of Amazon reviews used in this project
│   ├── summarized_review.csv          # Summarized reviews data for content-based filtering
│   └── vectorized_data.csv            # Vectorized review data used in model input
│
├── mds-learning-material/             # Supporting materials used during development
│
├── recommender_new.ipynb              # Jupyter notebook containing all model experiments and development
├── .gitignore                         # Git ignore file
├── LICENSE                            # License information
└── README.md                          # Project documentation (this file)
```

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Pandas
- Scikit-learn
- NumPy
- Hugging Face Transformers (for LLMs)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/llm-recommender-system.git
cd llm-recommender-system
```

2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the `data/` folder.

4. Run the notebook in Jupyter:

```bash
jupyter notebook recommender_new.ipynb
```

---

## Model Development

### 1. **Collaborative Filtering**

We employed collaborative filtering to predict a user’s rating of a book based on the ratings from similar users. This model is built using matrix factorization techniques like Singular Value Decomposition (SVD) to identify latent factors that influence user ratings.

**Key Metrics**:
- RMSE: **0.94**

### 2. **Content-Based Filtering with LLMs**

The content-based filtering component of the system uses embeddings generated from user reviews via LLMs. Each review is passed through a transformer-based model (like BERT), which converts the text into a dense vector. These vectors are then used to find similarities between books based on their reviews.

**Key Metrics**:
- RMSE: **0.69**

### 3. **Hybrid Model**

By combining collaborative filtering with the content-based model, we create a robust hybrid recommender system that balances both user behavior and content similarity. The hybrid model leverages the strengths of both approaches, offering personalized recommendations while also considering the semantic content of user reviews.

---

## Usage

Once set up, you can explore the `recommender_new.ipynb` notebook, which walks through:

- **Data loading and preprocessing**: Handling missing values, text cleaning, and tokenization.
- **LLM-based vectorization**: Using a pre-trained model to generate semantic vectors from reviews.
- **Collaborative filtering**: Building a matrix factorization model using SVD.
- **Content-based filtering**: Implementing review-based recommendations using vector similarity.
- **Hybrid approach**: Combining collaborative filtering and content-based filtering for enhanced performance.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contributors

- **Mohammad Norouzi** - [GitHub Profile](https://github.com/MoNorouzi23)
- **Atabak Alishiri** - [GitHub Profile](https://github.com/atabak-alishiri)


Feel free to contribute by submitting a pull request or opening an issue!
