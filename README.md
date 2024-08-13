# Airline-Reviews-Sentiment-Analysis
This repository contains code and report for the 'Airline-Reviews-Sentiment-Analysis project.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Technical Overview](#technical-overview)
4. [Results](#results)
5. [Future Improvements](#future-improvements)

## Project Overview

During the past years, I have fulfilled my passion for traveling by working as a Cabin Crew in Dubai. There is no sentence I have heard more often than: "Take care of the customers, because there is no airline without customers." A customer's journey begins when they book their ticket and does not conclude until they leave the airport at their destination. Throughout that process, every small detail can make the trip extraordinary or turn it into a disaster. Being on board and hearing firsthand thousands of different experiences has provided me with a deep understanding of customers' preferences. With the increasing competition in the aviation industry, leveraging this knowledge could significantly benefit a company’s reputation, increasing their number of travelers and revenue.

The aim of this project is to perform Exploratory Data Analysis (EDA) and Sentiment Analysis on airline reviews to understand customer satisfaction and identify key factors influencing their experiences. By analyzing the sentiment expressed in customer reviews and combining it with my real-life knowledge and experience, I expect to uncover insights that can help improve airline services and enhance customer satisfaction.

This project is divided into three main steps with future improvements in mind:

1. **Data Cleaning and Feature Engineering**
   - Clean and preprocess the dataset.
   - Use my experience to create new features from the existing data that could add value to the analysis.

2. **Exploratory Data Analysis (EDA) and Machine Learning**
   - Perform EDA on the available features.
   - Compare performance across different airlines.
   - Determine key features affecting customer loyalty.
   - Support these results with supervised machine learning models, utilizing the ‘Recommended’ and ‘Overall Rating’ values as targets.

3. **Natural Language Processing (NLP)**
   - Analyze passenger comments using NLP techniques.
   - Extract deeper insights and understand sentiments expressed.

## Data Description

The dataset used for this stage of the project is released on [Kaggle](https://www.kaggle.com/) under the MIT License. It contains 8,100 reviews left between March 2016 and March 2024.

### Data Details
- **Categorical Features:**
  - **Airline:** The name of the airline.
  - **Verified:** Indicates whether the review is verified.
  - **Type of Traveller:** Type of traveler (e.g., Business, Leisure).
  - **Class:** Travel class (e.g., Economy, Business).
  - **Recommended:** Whether the reviewer recommends the airline.

- **Numerical Ratings:**
  - **Seat Comfort**
  - **Staff Service**
  - **Food & Beverages**
  - **Inflight Entertainment**
  - **Value For Money**
  - **Overall Rating**

- **Date Features:**
  - **Review Date:** The date when the review was left.
  - **Month Flown:** The month when the flight occurred.

- **Text Features:**
  - **Title:** Title of the review.
  - **Reviews:** Full text of the review.
  - **Name:** Name of the reviewer.
  - **Route:** Route of the flight.

### Engineered Features (Included in the Processed Dataset)
- **Frequent Reviewer:** Categorizes reviewers into four different categories based on the number of verified reviews they have left.
- **Flight Month:** Extracts the month of the flight to analyze seasonal patterns.
- **Flight Year:** Extracts the year of the flight to analyze yearly trends and global changes (e.g., COVID-19).
- **Quick Review:** Computed by comparing the flight and review dates to identify reviews left soon after the flight.

## Technical Overview
### Libraries Used
- Pandas: For data manipulation and analysis
- NumPy: For numerical operations
- Scikit-learn: For machine learning models
- NLTK: For natural language processing
- Matplotlib & Seaborn: For data visualization

### Methods and Algorithms
- Data Cleaning: Correcting data types, dealing with data inconsistencies, text preprocessing
- Feature Engineering: Creating new features from existing data
- Sentiment Analysis: Using NLP techniques to analyze review text
- Model Building:
- Evaluation:

## Results
### Key Insights

### Model Performance

## Future Improvements
- Create my own dataset by scraping more reviews
- Experiment with deep learning models for sentiment analysis
- Incorporate more features related to customer demographics
