import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt

def cos_sim(x, y):
    return np.dot(x, y) / (norm(x) * norm(y))

def pos_sim(x, y):
    """calculate the "position similarity" of two strings

    compute the difference in distance to each word for every word in x

    this may be the most poorly written function of all time

    x: a bird eats a snake
    y: a snake eats a bird

    a: (bird, 3) (eats, 0) (a, 0) (snake, 3) => 1.5
    bird: (a, 0) (eats, 1) (a, 2) (snake, 0) => 0.75
    eats: (a, 0) (bird, 1) (a, 0) (snake, 1) => 0.5
    a: (a, 0) (bird, 1) (eats, 0) (snake, 1) => 0.5
    snake: (a, 0) (bird, 0) (eats, 1) (a, 2) => 0.75
    total average => 0.8
    """
    x_words = x.split(" ")
    y_words = y.split(" ")
    x_words_i = []
    y_words_i = []
    for i in range(len(x_words)):
        x_words_i.append((i, x_words[i]))
    for i in range(len(y_words)):
        y_words_i.append((i, y_words[i]))

    x_dists = []
    y_dists = []

    for (i, word) in x_words_i:
        new_dist = []
        for (j, word2) in x_words_i:
            if i != j:
                new_dist.append((abs(i - j), word2))
        x_dists.append((i, word, new_dist))

    for (i, word) in y_words_i:
        new_dist = []
        for (j, word2) in y_words_i:
            if i != j:
                new_dist.append((abs(i - j), word2))
        y_dists.append((i, word, new_dist, False))

    total = 0
    for (i, word, dist) in x_dists:
        y_ind = -1
        for (j, wordy, disty, used) in y_dists:
            if word == wordy and not used and y_ind == -1:
                y_ind = j
        if y_ind == -1:
            return -1
        (y_i, y_word, y_dist, y_used) = y_dists[y_ind]
        curr = 0

        # print(dist)
        # print(y_dist)

        for (d, w) in dist:
            min_diff = 10000
            min_ind = -1
            for (iy, (dy, wy)) in enumerate(y_dist):
                if w == wy:
                    diff = abs(d - dy)
                    if diff < min_diff:
                        min_diff = diff
                        min_ind = iy
            if min_diff == 10000:
                #print("strings not fully matched")
                return -1
            curr += min_diff
            del y_dist[min_ind]
        total += curr / (len(x_words) - 1)
        # print(total)
        y_dists[y_ind] = (y_i, y_word, y_dist, True)
    return total / (len(x_words) ** 2) # need some additional length penalty tbh


clip_embs = np.load("../data/clip_embs/text.npy")
print(clip_embs.shape)

cos_sims = []
for i in range(400):
    cos_sims.append(cos_sim(clip_embs[i][0], clip_embs[i][1]))
print(cos_sims)

bot5 = sorted(range(len(cos_sims)), key=lambda i: cos_sims[i])
print(bot5.index(321))
print(bot5[:5])

print(pos_sim("a bird eats a snake", "a snake eats a bird"))

f = open("../data/winoground_langdata.csv")
lines = f.readlines()
sents = []
pos_sims = []
sims_5 = []
num_5 = 0
for line in lines[1:]:
    s = line.split(",")
    sents.append((s[1], s[2]))
    if len(s[1].split(" ")) == 5:
        sims_5.append(pos_sim(s[1], s[2]))
        num_5 += 1
    else:
        sims_5.append(-1)
    pos_sims.append(pos_sim(s[1], s[2]))

top5 = sorted(range(len(pos_sims)), key=lambda i: pos_sims[i])[-5:]
print(top5)

toplen = sorted(range(len(sims_5)), key=lambda i: sims_5[i])
print(toplen.index(238) - (400 - num_5))
print(num_5)


print(cos_sims[345], pos_sims[345])
print(cos_sims[342], pos_sims[342])
print(cos_sims[0], pos_sims[0])
print(cos_sims[238], pos_sims[238])

# plt.scatter(pos_sims, cos_sims)
# plt.xlim([0, 0.4])
# plt.savefig("../outputs/sean_test.png")

print(cos_sims[0], cos_sims[5], cos_sims[364])
