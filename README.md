灾害应对管道
项目描述
针对此项目，建立一个模型来对灾难中的发布信息进行分类，共有36种信息分类类别。通过对信息进行分类分析，将相关的信息推送到相应的灾害救治机构，可以提高灾害应对的效率。
文件描述
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- Preparation
                |-- categories.csv
                |-- ETL Pipeline Preparation.ipynb
                |-- ETL_Preparation.db
                |-- messages.csv
                |-- ML Pipeline Preparation.ipynb
                |-- README
          |-- README
Installation
Must runing with Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries.

File Descriptions
App文件夹下的run.py文件用于使用模型对数据进行分类
Data文件夹中包含"DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" 用于数据的清洗与整合。
Models文件夹下包含"train_classifier.py" 用于建立机器学习模型。
Instructions
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

Licensing, Authors, Acknowledgements
Many thanks to Figure-8 for making this available to Udacity for training purposes. Special thanks to udacity for the training. Feel free to utilize the contents of this while citing me, udacity, and/or figure-8 accordingly.

NOTICE: Preparation folder is not necessary for this project to run.