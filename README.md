# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4.data文件夹中下的process_data.py文件，对disaster_categories.csv以及disaster_messages.csv进行数据的整理与合并，形成新的db数据库存入。

5.models文件夹下的train_classifier.py文件，对数据搭建机器学习管道，并对其进行网格搜索，计算得到效果较好的参数进行建模。存入model.pkl文件中。