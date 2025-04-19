
import pandas as pd
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk
nltk.download('punkt')
import nltk
nltk.download('averaged_perceptron_tagger')

# Read the data from the CSV file
df = pd.read_csv('phones.csv')

#input
sentence = "I like a average price and average size  "
#Dt A1: 40, 10, 15 = 0
#Dt A2: 40, 10, 5  = 0
#Dt A3: 40, 10, 1   = 0

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Part-of-speech tagging
pos_tags = pos_tag(tokens)

entities = {}
for i in range(len(pos_tags)):
    # Check for adjective + noun pairs
    if pos_tags[i][1] == 'JJ' and i+1 < len(pos_tags) and pos_tags[i+1][1] == 'NN':
        if pos_tags[i+1][0].lower() == 'size':
            entities['size'] = pos_tags[i][0].lower()
        elif pos_tags[i+1][0].lower() == 'battery':
            entities['battery'] = pos_tags[i][0].lower()
        elif pos_tags[i+1][0].lower() == 'price':
            entities['price'] = pos_tags[i][0].lower()
     # Check for noun + is + adjective pattern
    if i < len(pos_tags)-2 and pos_tags[i][1] == 'NN' and pos_tags[i+1][0].lower() == 'is' and pos_tags[i+2][1] == 'JJ':
        noun = pos_tags[i][0]
        adjective = pos_tags[i+2][0]
        entities[noun] = adjective
            
print(entities) 
num_elements = len(entities)
print(num_elements)
a = False
b = False
c = False

for x in entities:
    if x == 'price':
        a = True
    if x == 'size':
        b = True
    if x == 'battery':
        c = True

if a == True:
    if entities['price'] == "cheap":
        entities['price'] = df['price'].min()
    if entities['price'] == "expensive":
        entities['price'] = df['price'].max()
    if entities['price'] == "average":
        entities['price'] = df['price'].mean()
if b == True:
    if entities['size'] == "small": entities['size'] = df['size'].min()
    if entities['size'] == "big": entities['size'] = df['size'].max()
    if entities['size'] == "average": entities['size'] = df['size'].mean()
if c == True:
    if entities['battery'] == "short": entities['battery'] = df['battery'].min()
    if entities['battery'] == "long": entities['battery'] = df['battery'].max()
    if entities['battery'] == "average": entities['battery'] = df['battery'].mean()
print(entities)
f = 0
for x in entities:
    f += (df[x] - entities[x])**2
    
df['distance'] = f**0.5
    

# # Sort the dataset by distance
sorted_df = df.sort_values('distance')
print(sorted_df)

# # Output the top recommendation
top_phone = sorted_df.iloc[0]['names']
print("The recommended phone is:", top_phone)
