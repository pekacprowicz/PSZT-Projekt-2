import Classifier
from random import randint


class RandomForest:

    def __init__(self, size, data, rate, bootstrap=True):
        self.size = size
        self.trees = []
        self.rate = rate
        self.data = data

        for i in range(size):
            random_indices = []
            print("Cała liczebność zbioru: {0}  Liczebność podzbioru:   {1}".format(data.shape[0], int(data.shape[0] * rate)))

            if bootstrap:
                for _ in range(int(data.shape[0] * rate)):
                    random_indices.append(randint(0, data.shape[0] - 1))
            else:
                random_indices = range(0, data.shape[0])
            self.trees.append(Classifier.DecisionStumpClassifier(data.iloc[random_indices, :]))

    def estimate_class(self, data):
        rows = data.shape[0]
        table_ = []
        for tree in self.trees:
            results = tree.classify(data.iloc[:, range(0, len(data.columns)-1)])
            table_.append(results)
        result_trees = []

        for i in range(rows):
            result_trees.append([])
            for o in range(self.size):
                result_trees[i].append(table_[o][i])

        result_for_each_row = []
        for r_tree in result_trees:
            result_for_each_row.append(self.vote(r_tree))
        return result_for_each_row

    def add_new_tree(self):
        self.size += 1

        random_indices = []
        print("Cała liczebność zbioru: {0}  Liczebność podzbioru:   {1}".format(self.data.shape[0], int(self.data.shape[0] * self.rate)))
        for _ in range(int(self.data.shape[0] * self.rate)):
            random_indices.append(randint(0, self.data.shape[0] - 1))

        self.trees.append(Classifier.DecisionStumpClassifier(self.data.iloc[random_indices, :]))

    @staticmethod
    def vote(result_tree):
        class_result = {}
        for result in result_tree:
            if result in class_result:
                class_result[result] += 1
            else:
                class_result[result] = 1

        most_votes = max(class_result.values())
        final_vote = "start"

        for key, value in class_result.items():
            if value == most_votes:
                if final_vote == "start":
                    final_vote = key
                else:
                    final_vote = 0
        return final_vote
