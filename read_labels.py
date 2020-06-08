import numpy as np
import ujson as json
import sys


filename=sys.argv[1]
content = open(filename).readlines()

labels = []
for val in content:
    item = json.loads(val)
    labels.append(item["label"])

bettercounts = np.sum(labels, axis=0)

#print(labels)
labels = np.array(labels)
number_list = np.array([np.where(r==1)[0][0] for r in labels])
(unique, counts) = np.unique(number_list, return_counts=True)
frequencies = np.asarray((unique, bettercounts)).T
print(frequencies)
print(np.asarray((unique, np.round(max(bettercounts)/bettercounts))).T)
total = sum(bettercounts)
sample_total= sum(counts)
print("Total count: {}".format(total))
print(np.asarray((unique, bettercounts/total)).T)
print(bettercounts/len(labels))
