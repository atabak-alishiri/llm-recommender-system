# LLM-Powered Book Recommender System

## Project Overview

This repository contains a comprehensive implementation of a **Hybrid Book Recommender System**, combining **Collaborative Filtering** and **Content-Based Filtering** approaches, with the integration of **Large Language Models (LLMs)** for text processing. The primary goal is to enhance user experience by providing personalized book recommendations based on user ratings and review content.

![Llama Recommender System](img.jpg "Llama Recommender System")

### Key Features
- **Collaborative Filtering**: Recommends books by identifying patterns in user-item interactions (matrix factorization with SVD).
- **Content-Based Filtering with LLMs**: Uses LLM-generated text embeddings to recommend books based on the semantic similarity of reviews.
- **Scalable Summarization**: If sufficient computational power is available, review summaries are generated per book using the latest **LLaMA** model (Llama 3.2 1B), offering an additional level of granularity for the content-based model.

This hybrid system leverages the strengths of both collaborative and content-based techniques, providing a robust and scalable recommendation pipeline for book reviews.

---

## Key Achievements

- **Collaborative Filtering** RMSE: **0.94**
- **Content-Based Filtering** RMSE (LLM embeddings): **0.69**
- Optional **LLaMA**-based review summarization for more granular recommendations.
- Scalable data processing pipeline with dynamic dataset creation and storage.

---

## Dataset

The dataset used in this project is a subset of the **Amazon Book Reviews** dataset sourced from **Kaggle**. This dataset contains rich review information, including user ratings, review text, and product details, making it ideal for building both collaborative and content-based recommendation models.

### Dataset Details

- **Total Rows**: 3,105,370 (subsampled for this project)
- **Features**: 15
- **Key Columns**:
  - `customer_id`: Unique identifier for each customer.
  - `product_id`: Unique identifier for each book.
  - `product_title`: Title of the book.
  - `star_rating`: User rating (1-5 stars).
  - `review_headline` and `review_body`: User review content.
  - `helpful_votes` and `total_votes`: Feedback on review helpfulness.

The reviews are preprocessed, vectorized using LLMs, and utilized in the content-based filtering model to provide recommendations. Additionally, collaborative filtering predicts user ratings based on interaction patterns with other users.

Original dataset available on [Kaggle](https://www.kaggle.com/datasets/beaglelee/amazon-reviews-us-books-v1-02-tsv-zip).

---

## Main Notebook: `recommender_new.ipynb`

The primary focus of this project is the `recommender_new.ipynb` notebook, which outlines the development of the hybrid recommendation system. The notebook walks through the following key steps:

### 1. **Data Preprocessing**
   - Loading and cleaning the Amazon reviews dataset.
   - Grouping reviews by `product_id` and aggregating ratings to compute an average star rating per book.
   - Tokenization and vectorization of review text using LLMs, converting the textual content into dense semantic vectors for content-based filtering.

### 2. **Collaborative Filtering**
   - Using **Singular Value Decomposition (SVD)** to decompose the user-item interaction matrix.
   - Predicting user ratings for unseen books based on patterns of user behavior.
   - Evaluation using **Root Mean Squared Error (RMSE)** for accuracy, achieving a score of **0.94**.

### 3. **Content-Based Filtering**
   - Generating semantic embeddings from user reviews using a pre-trained **transformer model** (e.g., BERT or GPT).
   - Calculating similarities between books based on the textual content of reviews.
   - Recommendations are made by suggesting books that have similar content and sentiment to books the user has already rated.
   - Evaluation using RMSE, achieving a significant improvement over collaborative filtering with an RMSE of **0.69**.

### 4. **Hybrid Model**
   - Combining the collaborative filtering model with content-based filtering to create a hybrid recommender system.
   - The hybrid model leverages both user interaction patterns and semantic content from reviews to offer more personalized and accurate recommendations.

### Output:
   - A **dynamic pipeline** that saves processed datasets, including user-item interaction data and LLM-processed review embeddings, to the `data/` folder for efficient re-use.

---

## Optional Notebook: `llama_review_summarization.ipynb`

This notebook is an **additional feature** for environments with sufficient computational power. It demonstrates how to use **LLaMA** to summarize reviews for each book and generate more concise review text. These summaries can be used in place of or alongside the original review content in the content-based filtering model.

### Key Steps:
1. **Review Aggregation**: Aggregating multiple reviews per book.
2. **Summarization with LLaMA**: Using the LLaMA model to summarize the review text into shorter, more meaningful summaries.
3. **Embedding Generation**: Vectorizing the summarized reviews and saving them as `summarized_book_reviews_sampled_llama.csv`.
4. **Optional Use in Recommender**: These summarized reviews can be substituted into the content-based filtering pipeline for potentially faster processing and improved accuracy.

**Output**:
   - A file `data/summarized_book_reviews_sampled_llama.csv` that contains the summarized reviews, which can be used in place of full review text.

---

## Project Structure

```
├── data/                             # Contains dynamically saved data files
│   ├── amazon_reviews_subset.csv      # Subset of Amazon reviews used in this project
│   ├── summarized_book_reviews_sampled_llama.csv  # Summarized reviews by LLaMA
│   ├── vectorized_data.csv            # Vectorized review data used in model input
│   ├── summarized_review.csv            # Summarized review data used in model input
│
├── recommender_new.ipynb              # Main notebook for hybrid recommender development
├── llama_review_summarization.ipynb   # Optional notebook for summarizing reviews with LLaMA
├── config.json                        # Configuration file for Hugging Face token
├── environment.yml                    # Environment dependencies file
├── LICENSE                            # License information
└── README.md                          # Project documentation (this file)
```

---

## Installation and Setup

### Prerequisites

- Python 3.8+
- Conda (Anaconda or Miniconda)
- Hugging Face account for access to pre-trained models

### Installation Steps

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/llm-recommender-system.git
cd llm-recommender-system
```

2. **Create the environment using the YAML file**:

```bash
conda env create -f environment.yml
```

3. **Activate the environment**:

```bash
conda activate recommender_system
```

4. **Download the dataset** from Kaggle and place it in the `data/` folder.

5. **Add Hugging Face Token** to the `config.json` file:
   
   ```json
   {
     "hf_token": "your_token_from_hugging_face"
   }
   ```

6. **Run the notebooks**:

   - For building the hybrid recommender:
   
     ```bash
     jupyter notebook recommender_new.ipynb
     ```

   - For review summarization (optional):
   
     ```bash
     jupyter notebook llama_review_summarization.ipynb
     ```

---

## Model Development

### 1. **Collaborative Filtering**
   - Uses matrix factorization (SVD) to predict a user’s rating based on ratings from similar users.
   - RMSE: **0.94**

### 2. **Content-Based Filtering with LLMs**
   - LLM-generated embeddings are used to calculate similarities between books.
   - RMSE: **0.69**

### 3. **Hybrid Model**
   - The hybrid model combines collaborative filtering and content-based filtering for enhanced accuracy and personalization.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contributors

- **MoNorouzi23** - [GitHub Profile](https://github.com/MoNorouzi23)

Feel free to contribute by submitting a pull request or opening an issue!

---

## Contributors

- **Mohammad Norouzi** - [GitHub Profile](https://github.com/MoNorouzi23)
- **Atabak Alishiri** - [GitHub Profile](https://github.com/atabak-alishiri)


Feel free to contribute by submitting a pull request or opening an issue!
