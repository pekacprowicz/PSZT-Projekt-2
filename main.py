import pandas as pd
import RandomForest
import Classifier
import DataOperation
from numpy import arange    

def analyze_k(data):
    file = open("_analyze_.csv", "w")
    k_values = range(2, 21)

    percentage = []
    for l in range(10):
        data = data.sample(frac=1).reset_index(drop=True)
        for k in k_values:
            good, bad = 0, 0
            for i in range(k):
                data_learn, data_test = DataOperation.separate_data(DataOperation.bin_data(data, k), i)
                true_values = list(data_test.iloc[:, len(data_test.columns)-1])

                classifier = Classifier.DecisionStumpClassifier(data_learn)
                results = classifier.classify(data_test.iloc[:, range(0, len(data_test.columns)-1)])

                for o in range(len(true_values)):
                    if results[o] == true_values[o]:
                        good += 1
                    else:
                        bad += 1
                percentage.append(good/(good+bad)*100)
                print("Percentage: {}".format(good/(good+bad)*100))
                file.write(("{0};{1};{2};{3}%\n".format(l, k, i, good / (good + bad) * 100)))
    file.close()


def analyze_forest(data, rate):
    file = open("_final_bigger_forest_{0}.csv".format(rate), "w")
    start_trees = 5
    num_of_trees = start_trees
    percentage = []
    data_learn, data_test = DataOperation.separate_data(DataOperation.bin_data(data, 5), 2)
    true_values = list(data_test.iloc[:, len(data_test.columns)-1])

    classifier_ref = Classifier.DecisionStumpClassifier(data_learn)
    results_ref = classifier_ref.classify(data_test.iloc[:, range(0, len(data_test.columns)-1)])
    good_ref, bad_ref = 0, 0
    for o in range(len(true_values)):
        if results_ref[o] == true_values[o]:
            good_ref += 1
        else:
            bad_ref += 1
    file.write("Ref Bayes: {0}%".format(good_ref / (good_ref + bad_ref) * 100))
    print("Ref Bayes: {0}%".format(good_ref / (good_ref + bad_ref) * 100))
    classifier = RandomForest.RandomForest(start_trees, data_learn, rate)
    results = classifier.estimate_class(data_test)
    good, bad = 0, 0
    for i in range(len(true_values)):
        if results[i] == true_values[i]:
            good += 1
        else:
            bad += 1
    percentage.append(good / (good + bad) * 100)
    file.write("{0};{1}%\n".format(num_of_trees, good / (good + bad) * 100))

    for _ in range(30):
        for _ in range(5):
            classifier.add_new_tree()
            num_of_trees += 1
        results = classifier.estimate_class(data_test)
        good, bad = 0, 0
        for i in range(len(true_values)):
            if results[i] == true_values[i]:
                good += 1
            else:
                bad += 1
        percentage.append(good/(good+bad)*100)
        file.write("{0};{1}%\n".format(num_of_trees, good / (good + bad) * 100))
    file.close()


def analyze_forest2(data, data_test):
    file = open("analyze_forest2.csv", "w")
    start_trees = 2
    num_of_trees = start_trees
    k = 5
    percentage = []
    true_values = list(data_test.iloc[:, 0])
    classifier = RandomForest.RandomForest(start_trees, data, 1, False)
    results = classifier.estimate_class(data_test)
    good, bad = 0, 0
    for i in range(len(true_values)):
        if results[i] == true_values[i]:
            good += 1
        else:
            bad += 1
    percentage.append(good / (good + bad) * 100)
    file.write(("{0};{1}%\n".format(num_of_trees, good / (good + bad) * 100)))

    for _ in range(8):
        for _ in range(1):
            classifier.add_new_tree()
            num_of_trees += 1
        results = classifier.estimate_class(data_test)
        good, bad = 0, 0
        for i in range(len(true_values)):
            if results[i] == true_values[i]:
                good += 1
            else:
                bad += 1
        percentage.append(good/(good+bad)*100)
        file.write(("{0};{1}%\n".format(num_of_trees, good / (good + bad) * 100)))
    file.close()


def analyze_rate(data):
    file = open("test2_analyze_rate.csv", "w")
    rates = arange(1, 11, 1)
    percentage = []
    k_cross = 5
    for rate in rates:
        good, bad = 0, 0
        good_ref, bad_ref = 0, 0
        for k in range(k_cross):
            data_learn, data_test = DataOperation.separate_data(DataOperation.bin_data(data, k_cross), k)
            true_values = list(data_test.iloc[:, 0])

            classifier_ref = Classifier.DecisionStumpClassifier(data_learn)
            results_ref = classifier_ref.classify(data_test.iloc[:, range(1, len(data_test.columns))])

            for o in range(len(true_values)):
                if results_ref[o] == true_values[o]:
                    good_ref += 1
                else:
                    bad_ref += 1

            classifier = RandomForest.RandomForest(10, data_learn, rate)
            results = classifier.estimate_class(data_test)

            for i in range(len(true_values)):
                if results[i] == true_values[i]:
                    good += 1
                else:
                    bad += 1
        percentage.append(good / (good + bad) * 100)
        file.write("Ref Bayes: {0}%".format(good_ref / (good_ref + bad_ref) * 100))
        print("Ref Bayes: {0}%".format(good_ref / (good_ref + bad_ref) * 100))
        file.write(("{0};{1}%\n".format(rate, good / (good + bad) * 100)))
    file.close()


if __name__ == "__main__":
    data = pd.read_csv("car.data", dtype="category")
    # # data = pd.read_csv("agaricus-lepiota_short.csv", dtype="category")
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    data, data_test = DataOperation.separate_data(DataOperation.bin_data(shuffled_data, 10), 9)
    # analyze_k(shuffled_data)
    analyze_forest(shuffled_data, 0.01)
    # # analyze_forest(shuffled_data, 0.8)
    # # analyze_forest(shuffled_data, 1)
    # # analyze_forest(shuffled_data, 1.2)
    # # analyze_forest2(data, data_test)
    # # analyze_rate(shuffled_data)




