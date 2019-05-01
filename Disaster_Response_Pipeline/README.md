# Disaster Response Pipeline Project

This project is a part of Udacity Data Science Nanodegree.
Often during and after a disaster strikes, it is not easy to classify a response as legit or fake, and it comes on the way of quick help. Better classification of responses can help in faster aid and rescue. An efficient and user friendly app can be very crucial at these times. 



### Instructions to run this project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### User Interface (front-end, using flask):


