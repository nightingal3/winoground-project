from perturbtextorder import *
from perturbimage import *
from PIL import Image

text1 = "remarkable scene with a blue ball behind a green chair"
text2 = "The old man is slowly eating the sandwich and the pretty woman is calmly watching the television"
print("Evaluation Perturbations")
print("Original Text 1:",text1)
print("Shuffling Nouns and Adjectives:",nounadjshuf(text1))
print("Shuffling everything but Nouns and Adjectives:",nonnounadjshuf(text1))
print("Shuffling Trigrams:",trigramshuf(text1))
print("Shuffling Words in Trigrams:",wordtrigramshuf(text1))
print("Finetuning Perturbations")
print("Original Text 2:",text2)
print("Shuffling Nouns:",nounshuf(text2))
print("Shuffling Adjectives:",adjshuf(text2))
print("Shuffling Adverbs:",advshuf(text2))
print("Shuffling Noun Chunks:",nounchunkshuf(text2))

im = Image.open("87_1.png")
perturbim = RandomPatch(im,16)
perturbim.show()
