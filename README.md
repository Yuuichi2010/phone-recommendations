This repository contains the files for a Data Science project about recommender systems and machine learning in order to recommend phones for users using their historical data. 
The first project is building recommender systems using Content-Base Filtering. But after further work, we built the second project using Collaborative Filtering which can recommendations better. 
# CB Dataset: 
Our dataset contains 824 phone models, price, battery, and size.
reference of data: https://www.kaggle.com/datasets/anas123siddiqui/mobiles?fbclid=IwAR1LwZvB997RvLaBAQR59uc7UAuVI7gwL3PkO31VvkHKw-_nVwncOM0v3q8
# The process:
Firstly, in order to make a recommendations phone system, we start asking ourselves which dataset should be used, how to take user historical data, and recommend phone base in it. After some study, we find that we do it by finding the items-attribute dataset, taking user input and counting the distance of all the phones in the dataset with user input, sorting it, and recommending the smallest distance. 
Secondly, we start finding the dataset. The dataset which has the most phones and attributes was the dataset above. Then we start the cleaning process. All the phone which lacks information will be removed. We change the currency unit from Rupee to Dollar, change all the categorical data to numeric data, and choose 3 attributes of all to take the first step.
After our data is ready, we start making our system using Python language. At first, we will take user input as a sentence like: "I like an average price and big size". Then we use NLTK to find all the noun + is + adj and adj + noun patterns and put them in an object. After that, we started turning that adjective into a number. E.g. If the adj is average and the noun is price, we will use the mean() function in Python Libary to count the average price of all prices. 
Having the data describe the dream phone of the user, we will start calculating the distance of it with all the phones in the dataset, sort it, and take out the phone with the smallest distance.
At that point, we start sharing the project with other people and realize that this system is hard to evaluate with another system. All people will receive the same recommendation if they have the same input and the recommendation will lack diversity. So the Collaborative Filtering version was borns.
*This is the link in virtual workspace that you can run the algorithms: https://app.datacamp.com/workspace/w/5968582c-6b04-4914-beb3-1510b16e7bd4/edit
You can change the sentence to try that code. In this code, you input your dream phone and their will recommend the phone that match it most . 
# CF Dataset: 
Collaborative Filtering takes input data as a User-Rating matrix so we need to find another dataset. We choose the user-rating matrix from this dataset: https://www.kaggle.com/datasets/meirnizri/cellphones-recommendations?select=cellphones+data.csv
This data is already clean but the rating was out of 10 so we will need to scale it. Our work will follow these steps: prepare data, train the system, evaluate the system, and after that, we start to recommend things. But when recommending to new users, we need to add the user's historical information to the training set and recalculate it to make the recommendation.
Our main algorithm in the train system part is the matrix factorization technique using pure SVD with SGD which is used to optimize the system. And in the evaluation part, we use RMSE metrics. As a result, the RMSE is kinda high, about 2.08.
In this code, you input the phone id, your rating and their will recommend the phone that match it most .
# In the future: 
So in the future, we will try to optimize this system to make more accurate recommendations.
# P/S:
We were making this system for a Scientific paper at university. So feel free to read and leave some contributing comment.
