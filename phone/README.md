
# Phone Recommendation System

This repository contains a Phone Recommendation System developed using Machine Learning and Collaborative Filtering techniques. It leverages a trained model to recommend phone based on user preferences and uses Streamlit to provide a user-friendly web interface for generating recommendations.



## Project Overview
The primary goal of this project is to create a recommendation engine that suggests phone to users based on their reading history. The model uses collaborative filtering, a popular approach in recommendation systems that makes predictions about a user's interests by collecting preferences from many users.
## Key features
Collaborative Filtering: Utilizes user-item interactions to recommend phone with similar characteristics.
Streamlit Web Application: A simple and interactive interface where users can input a phone name and receive personalized recommendations.
## Setup Instructions

Clone the Repository:

```bash
  git clone https://github.com/Advitiyyaaa/Phone-recommender-system.git
  cd Phone-recommender-system
```
Create a Virtual Environment:

```bash
  conda create --prefix ./env python=3.7 -y
  conda activate ./env
```
Install Required Packages:
```bash
  pip install -r requirements.txt
```
Run the Streamlit App:
```bash
  streamlit run app.py

```
Navigate to Localhost: Open http://localhost:8501 in your browser to interact with the application.
