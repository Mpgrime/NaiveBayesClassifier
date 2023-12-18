import json  # deals with json files
import pandas as pd  # lets me frame this data
import time  # lets me measure how long shit takes
import string
import math # logs
import random # for holdout validation
import nltk #for lemmatizing

def calcOverAll(d):
    total = len(d.index)  # total amount of articles
   
    probs = pd.DataFrame(columns=["probability"], index=categories, dtype="float64")
    probs.fillna(0, inplace=True)
    for row in d.itertuples(name=None):  # basically the same as the word prob chart
        
        probs.loc[row[3], "probability"] += 1  # but with the categories instead
    for row in probs.itertuples(name=None):
        probs.loc[row[0], "probability"] = probs.loc[row[0], "probability"] / total
    probs = probs["probability"].tolist()
    return probs


def calcCate(w, p):
    # get the words from the text
    words = []
    w = w.lower()  # lowercase
    for j in w.split():
        j = j.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
        if ( not j.isdigit() and j not in stopwords):  # if it  isn't a number or a stopword
            words.append(lemma.lemmatize(j))  # add it to the list after lemmatizing it
    words = list(set(words))  # remove duplicates
    words.insert(0, 'overAll') # add overall to the columns
    # create df to hold category probabilities
    i = p.columns.values.tolist() #create list of categories
    probabilities = pd.DataFrame(index=i, columns=words) #index = categories, columns = each word
    # fill in each probability
    #fills in overall probability
    
    for row in probabilities.itertuples():
        for word in words:
            try:
                p.at[word, row[0]] #check that the word is in the model
            except:
                probabilities.drop(word, axis=1, inplace=True)  #if the word can't be found, get rid of it 
                words.remove(word)
            else:
                probabilities.at[row[0], word] = math.log(p.at[word, row[0]]) #find the probability of the word appearing in that category


    totals = [] #get the totals for each category
    total = 0
    for row in probabilities.itertuples(name=None):
        total = 0
        i = 1
        while i < len(row):
            total += row[i]
            i += 1
        totals.append(total)  

    probabilities['totals'] = totals # add them to the dataframe
    
    max = -1000000000000 #find the category with the largest total
    for elem in probabilities['totals']:
        if elem > max:
            max = elem
            category = probabilities.loc[probabilities['totals'] == elem].index[0]
    
    return category # return the category with the highest likelihood

#setting up lemmatizing
#code from https://stackoverflow.com/questions/24647400/what-is-the-best-stemming-method-in-python 
lemma = nltk.wordnet.WordNetLemmatizer()

# get stopwords into a list
initStart = time.time()
start = time.time()
print("--BUILDING MODEL--")
file = open("stopwords.txt", "r")  # opens the stopwords file
stopwords = []  # init list to hold stopwords
for line in file:  # iterates through each line in the file
    stopwords.append(line.replace("\n", ""))  # adds each line to the list
file.close()  # closes the file
finish = time.time()
print("Stopwords loaded in %s seconds" % (time.time() - start))

# get the data out of the json file
start = time.time()
file = open("News_Category_Dataset_v3.json", "r")  # opens the json file as file
testSrc = []  # init list to hold the majority of the data
modelSrc = []  # init list to hold data to test with
x = 0
for line in file:  # iterates through each line in the file
    j = random.randint(1,5)
    if j == 1: #roughly 20%
        testSrc.append(json.loads(line))  # add it to the test data
    else:
        modelSrc.append(json.loads(line))  # add it to the model data
file.close()  # closes the file
print("Data loaded in %s seconds" % (time.time() - start))

# get the data into proper dataframes
start = time.time()
model = pd.DataFrame.from_dict(modelSrc)  # converts model data into dataframe
test = pd.DataFrame.from_dict(testSrc)  # converts test data into dataframe
data = pd.read_json('News_Category_Dataset_v3.json', lines=True) # gets the whole data set

print("Converted to dataframes in %s seconds" % (time.time() - start))

# stopword removal code from https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe
# punctuation removal from https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
# remove the stopwords from headline
start = time.time()
model["headline"] = model["headline"].apply(lambda i: [word.lower() for word in i.split()])  # make all words lowercase
model["headline"] = model["headline"].apply(lambda i: [word.translate(str.maketrans("", "", string.punctuation)) for word in i])  # remove punctuation
model["headline"] = model["headline"].apply(lambda i: [word for word in i if not word.isdigit()])  # remove numbers
model["headline"] = model["headline"].apply(lambda i: [word for word in i if word not in stopwords])  # remove stopwords
model["headline"] = model["headline"].apply(lambda i: [lemma.lemmatize(word) for word in i])  # lemmatize
print("Stopwords removed from headlines in %s seconds" % (time.time() - start))

# remove the stopwords from description
start = time.time()
model["short_description"] = model["short_description"].apply(lambda i: [word.lower() for word in i.split()])
model["short_description"] = model["short_description"].apply(lambda i: [word.translate(str.maketrans("", "", string.punctuation)) for word in i])
model["short_description"] = model["short_description"].apply(lambda i: [word for word in i if not word.isdigit()])
model["short_description"] = model["short_description"].apply(lambda i: [word for word in i if word not in stopwords])
model["short_description"] = model["short_description"].apply(lambda i: [lemma.lemmatize(word) for word in i])  # lemmatize
print("Stopwords removed from descriptions in %s seconds" % (time.time() - start))

# list unique categories
start = time.time()
categories = pd.Index(data.category.unique())
print("Articles grouped by category in %s seconds" % (time.time() - start))
# get overall probability
start = time.time()
overAll = calcOverAll(model)
print("Overall probability calculated in %s seconds" % (time.time() - start))

# list unique words
start = time.time()
words = []
for i in model["short_description"]:
    for j in i:
        words.append(j)
for i in model["headline"]:
    for j in i:
        words.append(j)
words = pd.Index(list(set(words)))
print("Unique words listed in %s seconds" % (time.time() - start))

# create a dataframe of the unique words and categories
start = time.time()
modelP = pd.DataFrame(columns=categories, index=words, dtype="float64")
modelP.fillna(1, inplace=True)# laplace smoothing 
print("Word Chart created in %s seconds" % (time.time() - start))
# fill the probability dataframe
start = time.time()
print("Filling word chart totals")
i = 0
for row in model.itertuples(name=None):  # for every row in the source data
    for word in row[2]:  # for every word in the headline
        modelP.at[word, row[3]] += 1  # add one to the corresponding word/category in the prob table
    for word in row[4]:  # same for the description
        modelP.at[word, row[3]] += 1
    i += 1
    if i == 41905:
        print("25 percent done")
    if i == 83811:
        print("50 percent done")
    if i == 125716:
        print("75 percent done")
print("Word Chart totals filled in %s seconds" % (time.time() - start))

# remove weird row with blank index
modelP.drop(index="", inplace=True)  

# calculate the probabilities that each word appears in each category
start = time.time()
print("Calculating model probabilities")
j = 0
for row in modelP.itertuples(name=None):  # for each row
    # get the total amount of times that word appears
    total = 0
    i = 1
    while i < len(row):
        total += row[i]
        i += 1
    # get how often it appears per column (probability that word appears in each category)
    for col in modelP:
        modelP.at[row[0], col] = modelP.at[row[0], col] / total
    j += 1
    if j == 26962:
        print("25 percent done")
    if j == 53925:
        print("50 percent done")
    if j == 80887:
        print("75 percent done")
print("Word probabilities calculated in %s seconds" % (time.time() - start))

# add the overall probabilities to modelP
overall = pd.DataFrame([overAll], columns=modelP.columns, index=['overAll']) 
modelP = pd.concat([overall, modelP])
print("--MODEL FINISHED IN %s SECONDS--" % (time.time() - initStart))
# calculate which article belongs in which category
start = time.time()
print("--SORTING TEST DATA--")
modelProbs = pd.DataFrame(columns=["category"])
i = 0
for row in test.itertuples(name=None):
    modelProbs.loc[len(modelProbs.index)] = [calcCate((row[2] +' '+  row[4]), modelP)]
    i += 1
    if i == 10476:
        print("25 percent done")
    if i == 20952:
        print("50 percent done")
    if i == 31428:
        print("75 percent done")
print("Test data sorted in %s seconds" % (time.time() - start))

start = time.time()
print("--CALCULATING ACCURACY--")
correct = 0
incorrect = 0
accuracy = pd.DataFrame(columns=data.category.unique(), index = ['correct','incorrect', 'total', 'accuracy']) # create a dataframe to hold accuracy information (correct, total, and accuracy per category)
accuracy.fillna(0.0, inplace=True) # fill each elem with 0.0

# fill the chart
for i in modelProbs.itertuples(): # for every article
    if i[1] == test.loc[i[0], 'category']: # if the predicted category matched the actual category
        correct += 1 # increase the total correct score
        total += 1
        accuracy.loc['correct', i[1]] += 1 # increase the correct score in that category
        accuracy.loc['total', i[1]] += 1 # increase the total amount of predictions in that category
    else:
        total += 1 # increase the total 
        incorrect += 1 # increase the total incorrect
        accuracy.loc['incorrect', test.loc[i[0],'category']] += 1
        accuracy.loc['total', test.loc[i[0],'category']] += 1 # increase the total for that category
score = (correct/total)*100 # calc the overall accuracy
for category in accuracy: #calc the accuracy for each category
    accuracy.loc['accuracy', category] = ((accuracy.loc['correct', category] / accuracy.loc['total', category])*100)
    if accuracy.loc['accuracy', category] == 0.0: # if a categoryy had no predictions mark it as so
        accuracy.loc['accuracy', category] = 0
accuracy.fillna(0, inplace=True) # same here
print("Correct: ", correct)
print("Total: ", total)
print("--OVERALL--")
print("Accuracy was %s percent" % score)
print("--BY CATEGORY--")
print(accuracy.loc['accuracy'])
print("Accuracy calculated in %s seconds" % (time.time() - start))
print("Total time was %s seconds" % (time.time() - initStart))
