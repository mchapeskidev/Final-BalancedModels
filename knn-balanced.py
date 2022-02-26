import numpy as np
import pandas
import re


import seaborn as sbn
import matplotlib.pyplot as plt
import gensim.downloader as api


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



MAX_VOCAB = 5000
MAX_OUTPUT = 4


def clean_text(story_data):
    """
    This function served to clean the text by removing all punction, while also removing stopwords, leaving only the
    most important words to the meaning of the phrase.
    Regex code was sourced from the following site
    https://monkeylearn.com/blog/text-cleaning/
    :param story_data:
    :return: copy_story_data, the cleaned version of the text
    """
    copy_story_data = story_data
    for i in range(len(copy_story_data)):
        copy_story_data[i] = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", copy_story_data[i])
    for j in range(len(copy_story_data)):
        copy_story_data[j] = remove_stopwords(copy_story_data[j])

    return copy_story_data


def vectorise_text(s_text):
    cleaned_text = clean_text(s_text)
    w_model = api.load('word2vec-google-news-300')
    # 300 is the feature size, input dimensionality
    print("got here")
    w_vectors = []
    for synopsis in cleaned_text:
        valid_words = []
        for word in range(len(synopsis)):
            if synopsis[word] in w_model:
                valid_words.append(synopsis[word])
            else:
                pass
        w_vectors.append(np.mean(w_model[valid_words], axis=0))

    return w_vectors

"""
def predict_theme(story_train, genre_train):
    print("Starting Prediction...")
    X_train, X_test, y_train, y_test = train_test_split(story_train, genre_train, test_size=0.3)
    y_pred_list = []
    y_true_list = []
    pred_model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000)
    pred_model.fit(X_train, y_train)
    y_predicted = pred_model.predict(X_test)
    #y_pred_list.append(y_predicted)
    #y_true_list.append(genre_test)
    val_scores = cross_val_score(pred_model, X_test, y_test, cv=3, n_jobs=-1)
    acc_scores = accuracy_score(y_true=y_test, y_pred=y_predicted)
    return y_predicted, y_test, val_scores, acc_scores
"""
def predict_theme(story_train, genre_train):
    print("Starting Prediction...")
    X_train, X_test, y_train, y_test = train_test_split(story_train, genre_train, test_size=0.3)
    pred_model = KNeighborsClassifier(n_neighbors=129)
    pred_model.fit(X_train, y_train)
    y_predicted = pred_model.predict(X_test)
    acc_scores = accuracy_score(y_true=y_test, y_pred=y_predicted)

    return y_predicted, y_test, acc_scores


def preprocess_model():
    raw_data = pandas.read_csv("movie_data.txt", engine="python", sep=":::")
    raw_data.columns = ["film_id", "film_title", "genre", "description"]
    text_gen_dict_train = {}
    for idx in raw_data.index:
        if raw_data["genre"][idx] in text_gen_dict_train:
            text_gen_dict_train[raw_data["genre"][idx]] += 1
        else:
            text_gen_dict_train[raw_data["genre"][idx]] = 1

    #print(text_gen_dict_train)

    #raw_data = raw_data.groupby('genre')
    #raw_data = pd.DataFrame(raw_data.apply(lambda x: x.sample(raw_data.size().min()).reset_index(drop=True)))


    train_text = raw_data['description']
    train_text = train_text.to_numpy()
    train_theme = raw_data['genre']
    train_theme = train_theme.to_numpy()

    counter_dict = {" thriller ":0, " adult ":0, " drama ":0, " documentary ":0,
                    " comedy ":0, " crime ":0, " reality-tv ":0, " horror ":0,
                    " sport ":0, " animation ":0, " action ":0, " fantasy ":0,
                    " short ":0, " sci-fi ":0, " music ":0, " adventure ":0,
                    " talk-show ":0, " western ":0, " family ":0, " mystery ":0,
                    " history ":0, " news ":0, " biography ":0, " romance ":0,
                    " game-show ":0, " musical ":0, " war ":0}
    b_text_train = []
    b_gen_train = []
    for i in range(len(train_text)):
        if counter_dict[train_theme[i]] <= 1000:
            b_text_train.append(train_text[i])
            b_gen_train.append(train_theme[i])
            counter_dict[train_theme[i]] += 1
        else:
            pass
    """
    balance_model = RandomUnderSampler()
    train_text = train_text.reshape(-1,1)
    #train_theme = train_theme.reshape(-1, 1)
    balanced_train_text, balanced_train_theme = balance_model.fit_resample(train_text, train_theme)
    """
    balanced_train_text = np.array(b_text_train)
    balanced_train_theme = np.array(b_gen_train)

    #print(train_theme)
    train_raw_theme = balanced_train_theme
    # Do one-hot encoding on the theme-variable; as it never gets vectorised.
    lab_enc = LabelEncoder()
    train_theme = lab_enc.fit_transform(train_raw_theme)


    lab_enc2 = LabelEncoder()

    """
    plt.hist(x=balanced_train_theme, bins=list(range(28)))
    plt.title("Frequency of Various Genres in Balanced Dataset")
    plt.xticks(rotation=90)
    plt.xlabel("Genre")
    plt.ylabel("Frequency")
    plt.show()
    """

    train_text = vectorise_text(balanced_train_text)


    train_text = np.array(train_text)


    #print(test_text.shape)
    #train_text = train_text[:-14, :]
    #print(train_text.shape)

    """
    train_text = np.expand_dims(train_text, -1)  # new shape = (2340, 590, 1)
    test_text = np.expand_dims(test_text, -1)  # new shape = (2340, 590, 1)
    train_text = train_text.swapaxes(1,2)
    test_text = test_text.swapaxes(1, 2)
    """



    #print(balanced_train_text.shape)
    # 16851 entries
    #print(test_text.shape)

    return train_text, train_theme


def run_model():
    train_text, train_theme = preprocess_model()
    predicted_labels, true_labels, acc_scores = predict_theme(train_text, train_theme)
    print(predicted_labels)
    print("\n\n")
    print(true_labels)
    print("Accuracy Is: ", acc_scores)
    labs = list(range(27))
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labs)
    sbn.heatmap(conf_matrix)
    plt.title("Predicted Genre vs. True Genre \nusing K-Nearest Neighbours (k=129)\n Balanced Dataset")
    plt.show()



# run_model()
preprocess_model()