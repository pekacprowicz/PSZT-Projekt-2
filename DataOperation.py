import pandas
import math
import copy


def bin_data(data, bins):
    length = data.shape[0]
    length_bin = length / bins
    length_bin = math.ceil(length_bin)
    bins_range = []

    for i in range(bins-1):
        bins_range.append(range(length_bin*i, length_bin*(i+1)))

    bins_range.append(range(bins_range[len(bins_range)-1][len(bins_range[len(bins_range)-1])-1]+1, length))

    output_data = []

    for bin_range in bins_range:
        output_data.append(data.iloc[bin_range, :])

    return output_data


def separate_data(data, excluded):
    data_ = copy.deepcopy(data)
    data_to_test = data_[excluded]
    del data_[excluded]
    data_to_classifier = pandas.concat(data_)
    return data_to_classifier.reset_index(drop=True), data_to_test.reset_index(drop=True)