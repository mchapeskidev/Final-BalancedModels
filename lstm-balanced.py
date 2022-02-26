import numpy as np
import pandas
import seaborn as sbn
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import callbacks

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


MAX_VOCAB = 5000
MAX_OUTPUT = 4


def call_rounding(lst):
    for i in range(len(lst)):
        lst[i] = round(lst[i])
    return lst


def build_lstm_model(story_data):
    dictionary = TextVectorization(max_tokens=MAX_VOCAB, output_mode="int", output_sequence_length=MAX_OUTPUT)
    dictionary.adapt(story_data)
    model = Sequential()
    model.add(Input(shape=(1,), dtype="string"))
    model.add(dictionary)
    model.add(Embedding(input_dim=len(dictionary.get_vocabulary()), output_dim=32, mask_zero=True))
    model.add(Bidirectional(LSTM(32, kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))))
    model.add(BatchNormalization())
    model.add(Dense(100, activation="relu", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    #model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
    #model.add(Dropout(0.6))
    #model.add(BatchNormalization())
    model.add(Dense(40, activation="relu", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(28, activation="softmax"))
    model.compile(
        optimizer=optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model



def predict_theme(story_train,genre_train):
    # y_pred_list = []
    # y_true_list = []
    X_train, X_test, Y_train, Y_test = train_test_split(story_train, genre_train, test_size=0.3)
    pred_model = build_lstm_model(story_train)
    history = pred_model.fit(X_train, Y_train, validation_split=0.3, epochs=20, verbose=1)

    # Sourced from https://stackoverflow.com/a/56807595
    """
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("RNN Train/Validation Accuracy by Epoch\n Balanced Dataset")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper left")
    plt.show()
    """
    y_predicted = pred_model.predict(X_test)
    usable_metrics = pred_model.evaluate(X_test, Y_test)
    return y_predicted, Y_test, usable_metrics


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

    return balanced_train_text, train_theme


def run_model():
    train_text, train_theme= preprocess_model()
    predicted_labels, true_labels, overall_acc = predict_theme(train_text, train_theme)
    #print(predicted_labels)
    #print("\n\n")
    #print(true_labels)
    #for i in range(len(overall_acc)):
        #print("Loss, accuracy run" + str(i) + ": ", overall_acc)
    #print(predicted_labels.shape)
    #print(len(predicted_labels[0]))
    #print(np.max(predicted_labels[0]))
    return predicted_labels, true_labels


def create_output(done_predicted_labs, the_true_labs):

    corresponding_labels = []
    for i in done_predicted_labs:
        corresponding_labels.append(np.argmax(i))

    #print(type(done_predicted_labs))
    #print(done_predicted_labs[0])
    #print(type(the_true_labs))
    # print(done_predicted_labs.shape)
    #print(the_true_labs.shape)
    # demo_labs = list(range(27))
    acc_score = accuracy_score(y_true=the_true_labs, y_pred=corresponding_labels)
    print("SKLearn Accuracy: ", acc_score)
    conf_matrix = confusion_matrix(the_true_labs, corresponding_labels, labels=None)
    sbn.heatmap(conf_matrix)
    plt.title("Predicted Genre vs. True Genre using an RNN Deep Learning Model\n Balanced Dataset")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()






predicted_labs, true_labs = run_model()
create_output(predicted_labs, true_labs)
#preprocess_model()