# Udacity Data-Science Nanodegree Capstone Project - 'Sparkify'


## Table of Contents
<a href="section1">1. Installation 4</a>
2. Problem Definition
3. Motivation
4. Files
5. Result
6. Enhancements Potential
6. Licensing, Authors, and Acknowledgements


## Installation<A name="section1"></A>
This project is implemented in Python, using below Libraries-

Pyspark - for major data pre-processing and modelling
Pandas - for data pre-processing 
Matplotlib & Seaborn - for data visualization and exploration 


## Problem Definition 
We are provided with a user dataset consisting of their activity at different time stamps for a music service called 'sparkify' (Sparkify is a music streaming service just as Spotify and Pandora).Our task is to come up with several features and predict the churn of users. Being able to predict the churn, Sparkify can improve their retention rate by identifying who are the most vulnerable users early in the stage and improving their service.


## Motivation
Biggest motivation behind choosing this project was to develop my Big data skills and learn Spark. Also being able to know how to deploy the model on IBM Cloud.


## Files 
The Repo contains only the 'Sparkify.ipynb' notebook which is a result of working on 128MB 'mini_sparkify_event_data.json' dataset provided by Udacity.


## Results 
Logisitc Regression, Random Forest and Linear SVC were used as Models to predict user churn, optimizing the F1-Metric (which was found appropriate given the nature of the project). Random Forest model performed best. The result for same is published as part of blogpost [here]().


## Licensing, Authors, Acknowledgements
The work here is for educational purpose & is Open Source and can therefore be use by anyone for manipulation and help (feel free to use the code as you like). 

Credit for the project goes to [Udacity](https://www.udacity.com/) for all the help & support in completion of this project as part of the coursework. 
