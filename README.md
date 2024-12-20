# Time-series data and application to stock markets
## Navigation
- Task 1-4: `notebooks` directory
- Task 5.1,2: `deployment` directory
- Task 5.3: `airflow` directory
- Task 6: `report` directory

## Overview

This project explores the use of time-series data to forecast stock market trends and generate actionable insights. By analyzing Nasdaq and Vietnam stock data, I developed predictive models, trading signal generators, and portfolio simulations. The project culminated in the deployment of a stock prediction pipeline using advanced engineering workflows, including automation with Apache Airflow and deployment as a SaaS platform.

## Table of Contents

1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Folder Structure](#folder-structure)
4. [Project Workflow](#project-workflow)
    - [Task 1: Nasdaq Stock Price Prediction](#task-1-nasdaq-stock-price-prediction)
    - [Task 2: Vietnam Stock Price Prediction](#task-2-vietnam-stock-price-prediction)
    - [Task 3: Trading Signal Identification for Vietnam Market](#task-3-trading-signal-identification-for-vietnam-market)
    - [Task 4: Risk and Return Analysis Across Industries](#task-4-risk-and-return-analysis-across-industries)
    - [Task 5: Industry Standard for Deployment and Ease of Use](#task-5-industry-standard-for-deployment-and-ease-of-use)
5. [Lessons Learned](#lessons-learned)
6. [Future Improvements](#future-improvements)

## Introduction

The project focuses on leveraging time-series data for stock market predictions and decision-making. Using historical data, I built predictive models and automated workflows, transforming raw data into actionable insights. Each task aimed to address unique challenges, including preprocessing large datasets, deploying scalable APIs, and automating complex pipelines.

## Key Features

- **Predictive Models**: Developed using Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks.
- **Trading Signal Generator**: Identified buy/sell/hold signals using technical indicators like SMA and RSI.
- **Portfolio Simulation**: Analyzed risk and return across industries and visualized performance metrics.
- **SaaS Deployment**: Deployed stock prediction as a REST API and SaaS platform.
- **Automated Workflow**: Built an Apache Airflow DAG to orchestrate tasks for multiple stock tickers.

## Project Workflow
### Task 1: Nasdaq Stock Price Prediction

- **Goal**: Develop a predictive model for Nasdaq stock prices.
- **Process**:
  1. Preprocessed datasets, ensuring chronological splits for training, validation, and testing.
  2. Built a CNN model with multiple Conv1D layers for temporal pattern extraction.
  3. Expanded predictions to future days while addressing error propagation.
- **Outcome**: Achieved robust predictions with lessons on data quality and model selection.

### Task 2: Vietnam Stock Price Prediction

- **Goal**: Adapt predictive modeling for Vietnam's stock market.
- **Process**:
  1. Fetched historical stock data using the `vnstock` library.
  2. Designed custom pipelines to handle data limitations and volatility.
  3. Improved robustness with ensemble methods and feature engineering.
- **Outcome**: Validated the adaptability of predictive techniques across different markets.

### Task 3: Trading Signal Identification for Vietnam Market

- **Goal**: Generate actionable trading signals.
- **Process**:
  1. Built an LSTM-based model to classify signals (buy/sell/hold).
  2. Enhanced accuracy using technical indicators like SMA and RSI.
  3. Experimented with moving average crossover strategies.
- **Outcome**: Balanced model complexity with interpretability, delivering competitive results.

### Task 4: Risk and Return Analysis Across Industries

- **Goal**: Analyze risk-return metrics for various industries.
- **Process**:
  1. Collected and processed data for Food & Beverage, Personal Goods, and Technology sectors.
  2. Visualized expected returns vs. risks using scatter plots.
  3. Simulated portfolios based on SMA crossovers and trading volume.
- **Outcome**: Combined analytical rigor with visual storytelling for investment insights.

### Task 5: Industry Standard for Deployment and Ease of Use

#### Task 5.1: Model Deployment as API Services
- **Goal**: Provide predictions via a RESTful API.
- **Outcome**: Deployed Flask-based APIs with interactive endpoints and visual outputs.

#### Task 5.2: Transforming the Model into SaaS
- **Goal**: Democratize stock prediction through a user-friendly SaaS platform.
- **Outcome**: Delivered a seamless web experience hosted on [Heroku](https://stock-prediction-1b8ddee43da9.herokuapp.com/).

#### Task 5.3: Orchestrating a Complete Workflow
- **Goal**: Build an end-to-end automated pipeline.
- **Outcome**:
  - Designed an Airflow DAG with Task Groups for scalable orchestration.
  - Streamlined workflows for multiple stock tickers, reducing complexity.

## Lessons Learned

- **Data Quality**: Accurate preprocessing is foundational to reliable predictions.
- **Adaptability**: Customizing models for distinct market conditions enhances robustness.
- **Orchestration**: Automation through tools like Airflow saves time and minimizes manual intervention.
- **User-Centric Design**: Intuitive interfaces improve accessibility and engagement.
- **Resource Management**: Optimizing dependencies ensures efficient, manageable systems.

## Future Improvements

- **Enhanced BI Integration**: Addressing challenges in building a full pipeline to visualize insights on platforms like PowerBI or Superset.
- **GPU Optimization**: Accelerating training and prediction processes.
- **Advanced Feature Engineering**: Exploring additional technical indicators for better predictive power.

## Conclusion

This project was a journey of growth, blending data engineering, machine learning, and deployment workflows. While not all ambitions were realized, the successes and lessons learned provide a strong foundation for future explorations. The project reinforces the value of perseverance, adaptability, and continuous learning in tackling complex problems.
