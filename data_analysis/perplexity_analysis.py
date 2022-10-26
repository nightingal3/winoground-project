import math

import matplotlib.pyplot as plt
import statistics
import numpy as np

"""
parses a file of the format sentence 1, sentence 2, perplexity 1, perplexity 2

:param
filename: the path to the file

:returns
pairs: a list where each element is of the form ((sentence 1, perplexity 1), (sentence 2, perplexity 2))
"""
def parse_perplexities(filename, separator="\t"):
    f = open(filename)
    lines = f.readlines()
    pairs = []
    for line in lines:
        s = line.strip().split(separator)
        pairs.append(((s[0], float(s[2])), (s[1], float(s[3]))))
    return pairs

"""
computes summary statistics of a list and plots a histogram

:param
data: a list containing the data

:returns
stats: returns some statistics in the form of (mean, std, max index, max value, min index, min value)
"""
def summary_statistics(data, plot_type="hist", plot_path=None, log_scale=False, upper=None, bins=100):
    mean = sum(data) / len(data)
    std = statistics.pstdev(data)
    max_val = max(data)
    max_ind = data.index(max_val)
    min_val = min(data)
    min_ind = data.index(min_val)

    if plot_path is not None:
        if upper is not None:
            plot_data = [d for d in data if d < upper]
        else:
            plot_data = data
        if plot_type == "hist":
            plt.hist(plot_data, bins=bins)
            if log_scale:
                plt.yscale('log')
                y_title = "Log of frequencies"
            else:
                y_title = "frequencies"
            plt.ylabel(y_title)
            plt.savefig(plot_path)
            plt.clf()
        elif plot_type == "box":
            plt.boxplot(plot_data)
            plt.ylabel("difference in perplexity")
            plt.savefig(plot_path)
            plt.clf()

    return (mean, std, max_ind, max_val, min_ind, min_val)

pairs = parse_perplexities("outputs/perplexity/winoground_perplexities.txt")
abs_diff = []
for ((s1, p1), (s2, p2)) in pairs:
    abs_diff.append(abs(p1 - p2))
(mean, std, max_ind, max_val, min_ind, min_val) = summary_statistics(abs_diff, "outputs/perplexity/abs_diff.png", True)
summary_statistics(abs_diff, "hist", "outputs/perplexity/abs_diff_upper1000.png", True, 1000)
summary_statistics(abs_diff, "hist", "outputs/perplexity/abs_diff_upper100.png", True, 100, 20)
summary_statistics(abs_diff, "box", "outputs/perplexity/abs_diff_box.png")
summary_statistics(abs_diff, "box", "outputs/perplexity/abs_diff_box1000.png", upper=1000)
print("Avg diff: " + str(mean))
print("std: " + str(std))
print("Max diff: " + str(max_val))
print("Max diff pair:")
print(pairs[max_ind][0][0])
print(pairs[max_ind][1][0])
print("Min diff: " + str(min_val))
print("Min diff pair:")
print(pairs[min_ind][0][0])
print(pairs[min_ind][1][0])

abs_diff_np = np.array(abs_diff)
under_20 = (abs_diff_np < 20).sum()
print("Number of pairs with absolute difference <20: " + str(under_20))

for pair in pairs:
    if abs(pair[0][1] - pair[1][1]) > 1000:
        print(pair[0][0], pair[1][0])