import pickle

var = input("Enter the news to verify: ")
print("You entered: " + str(var))


# function to run for prediction
def detecting_fake_news(var):
    # retrieving the best model for prediction call
    load_model = pickle.load(open('svm_linearSVC.sav', 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    if prediction == 0:
        return print('Berita Palsu , \n', prob[0][1])
    else:
        return print('Berita Benar , \n', prob[0][1])


if __name__ == '__main__':
    detecting_fake_news(var)
