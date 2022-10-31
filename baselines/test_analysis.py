from analysis_util import WinogroundResult, WinogroundResultList

CLIP_results = WinogroundResultList("BLIP/BLIP_scores.txt")

# to check our hypothesis that the captions both prefer the same image most of the time
same_image = 0
diff_image = 0
correct_image = 0
same_cap = 0
diff_cap = 0
correct_cap = 0
for result in CLIP_results.results:
    (c0, c1, i0, i1) = result.get_comparisons()
    if c0 == c1:
        same_image += 1
    else:
        diff_image += 1
    if c0 == 0 and c1 == 1:
        correct_image += 1

    if i0 == i1:
        same_cap += 1
    else:
        diff_cap += 1
    if i0 == 0 and i1 == 1:
        correct_cap += 1
print("prefer same image: " + str(same_image))
print("prefer diff image: " + str(diff_image))
print("correct image: " + str(correct_image))
print("prefer same caption: " + str(same_cap))
print("prefer diff caption: " + str(diff_cap))
print("correct caption: " + str(correct_cap))

# is this true if the images are from the same series?
CLIP_results_series = CLIP_results.filter_tag(["Series"])
same_image = 0
diff_image = 0
correct_image = 0
same_cap = 0
diff_cap = 0
correct_cap = 0
for result in CLIP_results_series.results:
    (c0, c1, i0, i1) = result.get_comparisons()
    if c0 == c1:
        same_image += 1
    else:
        diff_image += 1
    if c0 == 0 and c1 == 1:
        correct_image += 1

    if i0 == i1:
        same_cap += 1
    else:
        diff_cap += 1
    if i0 == 0 and i1 == 1:
        correct_cap += 1
print("(series) prefer same image: " + str(same_image))
print("(series) prefer diff image: " + str(diff_image))
print("(series) correct image: " + str(correct_image))
print("(series) prefer same caption: " + str(same_cap))
print("(series) prefer diff caption: " + str(diff_cap))
print("(series) correct caption: " + str(correct_cap))