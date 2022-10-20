from analysis_util import WinogroundResult, WinogroundResultList

CLIP_results = WinogroundResultList("CLIP_scores.txt")

# to check our hypothesis that the captions both prefer the same image most of the time
same_image = 0
diff_image = 0
correct = 0
for result in CLIP_results.results:
    (c0, c1, i0, i1) = result.get_comparisons()
    if c0 == c1:
        same_image += 1
    else:
        diff_image += 1
    if c0 == 0 and c1 == 1:
        correct += 1
print("prefer same image: " + str(same_image))
print("prefer diff image: " + str(diff_image))
print("correct: " + str(correct))

# is this true if the images are from the same series?
CLIP_results_series = CLIP_results.filter_tag(["Series"])
same_image = 0
diff_image = 0
correct = 0
for result in CLIP_results_series.results:
    (c0, c1, i0, i1) = result.get_comparisons()
    if c0 == c1:
        same_image += 1
    else:
        diff_image += 1
    if c0 == 0 and c1 == 1:
        correct += 1
print("(series) prefer same image: " + str(same_image))
print("(series) prefer diff image: " + str(diff_image))
print("(series) correct: " + str(correct))