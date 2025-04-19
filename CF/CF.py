import random
import operator
import numpy as np
import pandas as pd
from scipy import sparse
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.dataset import DatasetAutoFolds
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from surprise import Dataset
from surprise import Reader
from surprise import Trainset
from surprise import accuracy
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt

def input_data(): 
    df = pd.read_csv('cellphones ratings.csv', skiprows=lambda i: i>0 and random.random()==1.0)
    return df

def print_number_data(df):
    print(len(df))
    print(len(df['rating'].unique().tolist()))
    print(len(df['user_id'].unique().tolist()))
    print(len(df['cellphone_id'].unique().tolist()))

def scale_rating(df):
    reader = Reader(rating_scale=(0,10)) # rating scale range
    data = Dataset.load_from_df(df[['user_id', 'cellphone_id', 'rating']], reader)
    return data


def split_data(data):
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    return trainset, testset


def print_train_and_test(n, trainset, testset):

    for uid, iid, rating in itertools.islice(trainset.all_ratings(), n):
      print(f"User {uid} rated item {iid} with a rating of {rating}")

    for uid, iid, rating in testset[:n]:
      print(f"User {uid} rated item {iid} with a rating of {rating}")

def SVDalgo(trainset):
    algo = SVD(n_factors=50, n_epochs=30, lr_all=0.01, reg_all=0.04)
    algo.fit(trainset)
    return algo

def predic(algo, testset):
    predictions = algo.test(testset)
    return predictions

def evaluate(predictions):
    #count rmse
    accuracy.rmse(predictions)
    true_ratings = [pred.r_ui for pred in predictions]
    est_ratings = [pred.est for pred in predictions]
    #take all id from the testset 
    uids = [pred.uid for pred in predictions]
    return uids

def map_id_model(uids):
    #remove all the dup user ID
    users=list(set(uids))
    #map phone id and model
    c_data=pd.read_csv('cellphones data.csv')
    mapping = dict(zip(c_data['cellphone_id'],c_data['model']))
    return users, mapping

def recommendation(trainset, users, mapping, algo):
    items = trainset.build_anti_testset()
    for user in users[0:5]:
        user_items = list(filter(lambda x: x[0] == user, items))
        print(user)
        # generate recommendation
        recommendations = algo.test(user_items)
        recommendations.sort(key=operator.itemgetter(3), reverse=True)
        print(f"User {user} recommendations:")
        for r in recommendations[0:5]:
            print(f" [Title] {mapping[r[1]]}, [Estimated Rating] {round(r[3],3)}")


def add_data_to_trainset(trainset, new_data):
    # create a copy of the original trainset and put in n_trainset
    n_trainset = trainset.build_testset()

    # add new_data to n_trainset
    n_trainset.append(tuple(new_data))

    # create a new trainset object from the updated n_trainset
    reader = Reader(rating_scale=(1, 10))
    new_trainset = Dataset.load_from_df(pd.DataFrame(n_trainset), reader).build_full_trainset()

    return new_trainset


def evaluate_current_sys():
    print_number_data(input_data())
    data = scale_rating(input_data())
    trainset, testset = split_data(data)
    print_train_and_test(10,trainset,testset )
    algo = SVDalgo(trainset)
    predictions = predic(algo, testset)
    evaluate(predictions)



def take_user_input(trainset):
    #print all data in the cellphones data
    cf = pd.read_csv('cellphones data.csv')
    print("This is all the phone we have in dataset")
    for index, row in cf.iterrows():
        print(row['cellphone_id'], row['model'])

    #take new user historical rating and add that data to trainset
    n = int(input("Please number of phone you already buy and rate it: "))
    for i in range(n):
        id = int(input("Enter the cellphone_id: "))
        rate = int(input("Enter your rating: "))
        new_row = (259, id, rate)
        trainset = add_data_to_trainset(trainset, new_row)
    return trainset

def new_recommend_old_data():
    #first scale_rating
    data = scale_rating(input_data())
    #then split data
    trainset, testset = split_data(data)
    #print n data in trainset and testset
    print_train_and_test(10,trainset,testset)
    #use algorithm to predict rating
    algo = SVDalgo(trainset)
    predictions = predic(algo, testset)
    #count rmse ,evaluate current system, take all user id from the testset
    uids = evaluate(predictions)
    #mapping phone id and model, remove all the dup user ID
    users, mapping = map_id_model(uids)
    #generate recommendations
    recommendation(trainset, users, mapping, algo)

def recommendation_for_new_users():
    #take user_rating matrix and split like normal
    data = scale_rating(input_data())
    trainset, testset = split_data(data)
    #add historical users rating to train set
    trainset = take_user_input(trainset)
    #use SVD algo in the new trainset
    algo = SVDalgo(trainset)
    #this is user id which default given by system
    user = 259
    #map all phone id and model in the cellphones data set
    c_data=pd.read_csv('cellphones data.csv')
    mapping = dict(zip(c_data['cellphone_id'],c_data['model']))
    #take all items which people haven't rated yet
    items = trainset.build_anti_testset()
    #filtering to take all items which new user haven't rated yet
    user_items = list(filter(lambda x: x[0] == user, items))
    # generate recommendation for that user
    recommendations = algo.test(user_items)
    recommendations.sort(key=operator.itemgetter(3), reverse=True)
    print(f"User {user} recommendations:")
    for r in recommendations[0:5]:
        print(f" [Title] {mapping[r[1]]}, [Estimated Rating] {round(r[3],3)}")

def print_number():
    #print the number of all data in the dataset
    df = pd.read_csv('cellphones ratings.csv', skiprows=lambda i: i>0 and random.random()==1.0)
    print("total number of rating: ",len(df) )
    print("total number of user: ", len(df['user_id'].unique().tolist()))
    print("total number of cellphone: ",len(df['cellphone_id'].unique().tolist()))
    trainset, testset = train_test_split(scale_rating(df), test_size=0.2, random_state=42)
    print()
    #print the number of all data in the trainset

    num_ratings_trainset = trainset.n_ratings
    num_ratings_trainset = trainset.n_ratings
    num_users_trainset = trainset.n_users
    num_items_trainset = trainset.n_items
    print("Number of ratings in trainset:", num_ratings_trainset)
    print("Number of users in trainset:", num_users_trainset)
    print("Number of cellphone in trainset:", num_items_trainset)
    print()
    #print the number of all data in the testset

    testset = pd.DataFrame(testset, columns=['user_id', 'cellphone_id', 'rating'])
    print("Number of ratings in testset:", len(testset))
    print("Number of users in testset:", len(testset['user_id'].unique()))
    print("Number of cellphone in testset:", len(testset['cellphone_id'].unique()))

def draw_chart(): #draw RMSE chart in k runs 
    avg_rmse_values = []
    armse_values =[]
    i_values = list(range(0,50))

    for k in range(0, 50):
        data = scale_rating(input_data())
        trainset, testset = split_data(data)
        algo = SVD(n_factors= 50, n_epochs= 30, lr_all=0.01, reg_all=0.04)
        algo.fit(trainset)
        predictions = predic(algo, testset)
        rmse = accuracy.rmse(predictions)
        armse = rmse
        #put all the rmse in list variables
        armse_values.append(armse)
    # draw chart using that list
    plt.plot(i_values, armse_values)
    plt.xlabel('number of algorithm runs')
    plt.ylabel('RMSE')
    plt.title('Relationship between number of algorithm runs and RMSE')
    plt.show()

    


if __name__ == "__main__":
    new_recommend_old_data()
    #recommendation_for_new_users()
    #evaluate_current_sys()

