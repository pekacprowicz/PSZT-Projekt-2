from abc import ABC, abstractmethod
import math
import operator
import random


class Classifier(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def classify(self, input_data):
        pass


class DecisionStumpClassifier(Classifier):

    def __init__(self, data_to_learn):
        self.data = data_to_learn
        self.numberOfAttributes = len(self.data.columns)
        self.listOfAttributes = list(self.data.columns)
        self.numberOfUniqueValues, self.values_in_each_attribute = self.extract_unique_values()
        self.classifying_attribute, self.decision_stump = self.calculate_classifier()

    def extract_unique_values(self):
        calculated_number = []
        dict_of_values = {}
        for attribute in self.listOfAttributes:
            temp_cat = self.data[attribute]
            calculated_number.append(len(temp_cat.unique()))
            dict_of_values[attribute] = list(temp_cat.unique())
        return calculated_number, dict_of_values

    def calculate_attribute_entropy(self, attribute):
        possible_attribute_values = self.values_in_each_attribute.get(attribute)
        
        attribute_entropy = 0
        for attribute_value in possible_attribute_values:
            dataset_after_division = self.data[self.data[attribute] == attribute_value]
            
            class_attribute = self.listOfAttributes[self.numberOfAttributes-1]
            class_values = self.values_in_each_attribute.get(class_attribute)

            counted_class_values_dict = dict(dataset_after_division[class_attribute].value_counts())
            dataset_entropy = 0
            for class_value in class_values:
                if counted_class_values_dict.get(class_value) == 0:
                    dataset_entropy += float('-inf')
                else:
                    dataset_entropy += -(counted_class_values_dict.get(class_value)*math.log(counted_class_values_dict.get(class_value)))
            
            attribute_entropy += (dataset_after_division.shape[0] / self.data.shape[0]) * dataset_entropy
            
        return attribute_entropy

    def calculate_classifier(self):
        decision_stump = dict()

        class_attribute = self.listOfAttributes[self.numberOfAttributes-1]
        class_values = self.values_in_each_attribute.get(class_attribute)

        counted_class_values_dict = dict(self.data[class_attribute].value_counts())

        dataset_entropy = 0
        for class_value in class_values:
            dataset_entropy += -(counted_class_values_dict.get(class_value)*math.log(counted_class_values_dict.get(class_value)))

        max_inf_gain = float('-inf')
        attribute_to_split = None
        attributes_with_inf_gain = list()
        for attribute in self.listOfAttributes[0:self.numberOfAttributes-2]:
            attribute_entropy = self.calculate_attribute_entropy(attribute)
            information_gain = dataset_entropy - attribute_entropy
            if information_gain > max_inf_gain:
                attribute_to_split = attribute
                max_inf_gain = information_gain      
            if information_gain == float('inf'):
                attributes_with_inf_gain.append(attribute)

        if max_inf_gain == float('inf'):
            attribute_to_split = random.choice(attributes_with_inf_gain)

        for split_attribute_value in self.values_in_each_attribute.get(attribute_to_split):
            splitted_dataset = self.data[self.data[attribute_to_split] == split_attribute_value]
            counted_class_values_dict = dict(splitted_dataset[class_attribute].value_counts())
            decision_stump[split_attribute_value] = max(counted_class_values_dict.items(), key=operator.itemgetter(1))[0]
        print(f"Attribute selected to split: {attribute_to_split}")
        return attribute_to_split, decision_stump

    def classify(self, input_data):
        result = []
        list_of_attributes = list(input_data.columns)
        try:
            if self.classifying_attribute not in list_of_attributes:
                print(f"ClAttr: {self.classifying_attribute} / {list_of_attributes}")
                raise WrongTestData
        except WrongTestData:
            print("Provided test data is not valid")

        minimized_input_data = input_data.loc[:, [self.classifying_attribute]]
        for row in range(input_data.shape[0]):
            attribute_value = minimized_input_data.iloc[row, 0]
            try:
                result.append(self.decision_stump[attribute_value])
            except KeyError:
                result.append(None)
        
        print("Classify done job")
        return result


class WrongTestData(Exception):
    pass
