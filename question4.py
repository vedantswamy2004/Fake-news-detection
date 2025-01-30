import math
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
#genai-Used to get the vectorizer

vec = TfidfVectorizer()
def load_data():
    """""
    params:None
    Output: Returns: 
            data1 : List of lines in combined file, 
            labels: LIst of output for all lines in the file
             vec: This is the vectorizer which helps us transform it
             train: This is the vectorizer input just for the train
             val: This is just used for validation
             X_test:This is the testing input
             Y_val:This is the validation output
             Y_train:This is the training output
             
    """""
    with open('clean_fake.txt', 'r') as f1:
        lines1 = f1.readlines()

    # This will be the entire list of outputs
    labels = [0] * len(lines1)

    with open('clean_real.txt', 'r') as f2:
        lines2 = f2.readlines()

    #Concatenating both lists together
    labels.extend([1] * len(lines2))

    #This is the combined new file to get the combined data
    file_comb = "clean_fake_real.txt"

    #Opening the file
    with open(file_comb, 'w') as f3:
        with open('clean_fake.txt', 'r') as f1:
            f3.write(f1.read())
        with open('clean_real.txt', 'r') as f2:
            f3.write(f2.read())
    f4 = open(file_comb, 'r')

    #genai-Used to combine two files together
    with open('clean_fake_real.txt', 'r') as f5:
        data1 = f5.readlines()

    #Converting it into numpy array
    data = np.array(data1)

    data = vec.fit_transform(data)


    #genai -Used to understand how to use train_test_split
    train, X_temp, Y_train, Y_temp = train_test_split(data, labels, train_size=0.7, test_size=0.3, random_state = 42)

    val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, train_size=0.5, test_size=0.5, random_state = 42)

    return data1, labels, vec, train, val, X_test, Y_train, Y_val, Y_test
#Using the Decision Tree Classifier
data1, labels, vec, train, val, X_test, Y_train, Y_val, Y_test = load_data()

# We are starting to plot the accuracy vs max_depth
plt.xlabel('accuracy')
plt.ylabel('max_depth')


# Function to gie us the best accuracy
def select_model(max_depths, criterions):
    """""
    Input: 
         max_depths: List of depths
         criterions: List of criterions
    Output:
        max_model: Model with the best accuracy
        accuracy: Accuracy of all models
        dep: The depths with the corresponding accuracies
    """""
    #The grid of parameters
    parameter_grid = {'max_depth': max_depths, 'criterion': criterions}


    accuracy = []
    dep = []
    models = []
    max_acc = 0

    #For loop to go through all the parameters
    for k in parameter_grid['criterion']:
        for i in parameter_grid['max_depth']:

            model = DecisionTreeClassifier(max_depth=i, criterion=k, random_state=42)

            #Fitting our model
            model.fit(train, Y_train)
            models.append(model)

            #Prediction of Model
            y_pred = model.predict(val)

            #checking the accuracy of our model
            tru = 0
            for q in range(len(y_pred)):
                if y_pred[q] == Y_val[q]:
                    tru += 1
            acc = tru/len(y_pred)

            if acc >= max_acc:
                max_model = model
                max_acc = acc


            if k != "entropy":
                print(f"depth:{i}, criterion:{k}, accuracy:{acc}")
                accuracy.append(acc)
                dep.append(i)
            else:
                print(f"depth:{i}, criterion:{"information gain"}, accuracy:{acc}")
                accuracy.append(acc)
                dep.append(i)

        #Plotting the figure
        plt.figure(1)
        plt.plot(dep, accuracy)
        plt.ylabel('accuracy')
        plt.xlabel('max_depth')
        plt.title(k)
        plt.show()
        accuracy = []
        dep = []

    return max_model, accuracy, dep

#Testing the max_depths function
max_depths = [5,6,7,8,9]
criterions = ['entropy', 'gini', 'log_loss']

feature_names = vec.get_feature_names_out()
model, accuracy, dep = select_model(max_depths, criterions)

text_representation = tree.export_text(model, feature_names=feature_names)


plt.figure(figsize=(10,6))

#Plotting our trees
tree.plot_tree(model, max_depth = 2,
                   proportion = True, feature_names = feature_names, filled = True, class_names = ['Fake', 'Real'])

plt.show()

#Function to get the word from the index
def index_to_word(index , vocab):
    """""
    Input:
        index: index  in vectorizer
        vocab: vocabulary
    Ouput:
        Word
    """""


    vocab = {index:word for word, index in vocab.items()}
    return vocab.get(index)


def compute_information_gain(x,Y, train):
    """""
    Input: 
        x: This is the word
        Y: These are the labels
        train: This is the training input
    """""


    count = 0
    sum_true = 0

    #Computing the entropy for Y
    for q in Y:
        if q == 1:
            sum_true += 1

    #We are just computing the entropy of Y
    avg1 = sum_true/len(Y)
    avg2 = 1-avg1
    cross_y = -avg1 * math.log(avg1, 2) - avg2 * math.log(avg2, 2)

    #Computing the list of words in which it occurs in

    word_occur = []


    for j in range(len(train)):

        #The sentence has the word
        if (x in train[j].split()):
            word_occur.append(j)

    #We check all the labels in word_occur to check if the label is 1
    for ind in word_occur:
        if Y[ind] == 1:
            count += 1

    #Computing entropy of y_given_x
    if len(word_occur) != 0 and count != 0 and count != len(word_occur):

        #Avg1 is the probability of y given x
        new_avg1 = count / len(word_occur)

        #Avg2 is the probability of not y given x
        new_avg2 = 1 - new_avg1

        #The entropy of y given x
        cross_y_x_given = -new_avg1 * math.log(new_avg1, 2) - new_avg2 * math.log(new_avg2, 2)

    if count == 0 or count == len(word_occur):
        cross_y_x_given = 0

    #We repeat the above procedure for word not in the sentence
    word_not_occur = []

    for j in range(len(train)):
        if j not in word_occur:
            word_not_occur.append(j)

    count2 = 0

    #We compute the conditional entropy now

    for ind in word_not_occur:
        if Y[ind] == 1:
            count2 += 1
    if len(word_not_occur) == 0:
        return 0

    # Avg1 is the probability of y given not x
    new_avg1 = count2 / len(word_not_occur)

    # Avg2 is the probability of not y given not x
    new_avg2 = 1 - new_avg1


    if count2 != 0 and new_avg2 != 0:
        # The conditional entropy of y given not x
        cross_y_x_not_given = -new_avg1 * math.log(new_avg1, 2) - new_avg2 * math.log(new_avg2, 2)
    if count2 == 0 or count2 == len(word_not_occur):
        cross_y_x_not_given = 0

    cross_avg = (len(word_occur) * cross_y_x_given + cross_y_x_not_given * len(word_not_occur))/len(Y)

    #We return the actual entropy

    return cross_y - cross_avg

tree = model.tree_
root_node = 0
vocab = vec.vocabulary_


#Printing the information gain for the root node
print(f"word is {index_to_word(tree.feature[root_node], vocab)}")
word = index_to_word(tree.feature[root_node], vocab)


train_new, X_temp, Y_train_new, Y_temp = train_test_split(data1, labels, train_size=0.7, test_size=0.3, random_state = 42)

print(f"info gain of {index_to_word(tree.feature[root_node], vocab)} is {compute_information_gain(word,Y_train_new, train_new)}")
print(f"the information gain the is {compute_information_gain("the",Y_train_new, train_new)}")
print(f"the information gain trumps is {compute_information_gain("trumps", Y_train_new, train_new)}")
print(f"the information gain donald is {compute_information_gain("donald", Y_train_new, train_new)}")
print(f"the information gain america is {compute_information_gain("america", Y_train_new, train_new)}")
print(f"the information gain hillary is {compute_information_gain("hillary", Y_train_new, train_new)}")
