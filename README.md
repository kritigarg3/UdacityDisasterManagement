## UdacityDisasterManagement


## **Installation**

The files in this repository run on an anaconda installed Python environment. The .py files can be run on the terminal. Installed scikit-learn==0.20 for model evaluation:

## **Project Motivation**

For this project, the motivation was to develop a predictive supervised model which can classify text messages received during a disaster into various categories so that one is able to send the messages to an appropriate disaster relief agency. It also includes an application which enables one to input a new message and get the accurate classification results and provides visualizations of the data.

## **File Descriptions**

The datasets for this project (messages.csv and the categries.csv) were provided by Figure 8 to Udacity. There are 3 notebooks available here to showcase work.

**Message.csv** : includes raw messages received during disaster, the message id, the genre (whether it is social media, news, or direct text messages), and the various categories.

**Categories.csv:** contains the id and the different categories the message belongs to. A message can belong to multiple categories.

**process\_data.py:** It includes an ETL pipeline which takes in the raw datasets, cleans the data and returns a SQLite table.

**train\_classifier.py:** It contains a Machine Learning pipeline which takes the SQLite table as the input, cleans the data, builds a model and trains and evaluates the data to get the predictions.

**run.py:** Takes the results of &#39;process\_data.py&#39;, and &#39;train\_classifier.py&#39; and creates a web application where one types in any message during disaster and a category is generated for the same.

## **Results**

The main findings of the code can be found in the repository.

## **Licensing, Authors, Acknowledgements**

Must give credit to Figure 8 and Udacity for the data. Feel free to use the code here as you would like!
