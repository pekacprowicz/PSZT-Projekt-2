import pandas as pd
import RandomForest
import DataOperation


def analyze_forest(data, rate):
    file = open("_final_bigger_forest_{0}.csv".format(rate), "w")
    start_trees = 5
    num_of_trees = start_trees
    percentage = []
    data_learn, data_test = DataOperation.separate_data(DataOperation.bin_data(data, 5), 2)
    true_values = list(data_test.iloc[:, len(data_test.columns)-1])
    classifier = RandomForest.RandomForest(start_trees, data_learn, rate)

    for _ in range(30):
        results = classifier.estimate_class(data_test)
        good, bad = 0, 0
        for i in range(len(true_values)):
            if results[i] == true_values[i]:
                good += 1
            else:
                bad += 1
        percentage.append(good/(good+bad)*100)
        file.write("{0};{1}%\n".format(num_of_trees, good / (good + bad) * 100))

        for _ in range(5):
            classifier.add_new_tree()
            num_of_trees += 1
    file.close()


if __name__ == "__main__":
    raw_data = pd.read_csv("car.data", dtype="category")
    shuffled_data = raw_data.sample(frac=1).reset_index(drop=True)
    analyze_forest(shuffled_data, 0.1)
