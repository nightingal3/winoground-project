import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")
text = nlp("remarkable scene with a blue ball behind a green chair")
newtext = "remarkable scene with a blue ball behind a green chair".split()
print("Original Text: "+" ".join(newtext))
perturbation1 = newtext.copy()
adjectives = []
nouns = []
others = []
for token in text:
    if token.pos_=="ADJ":
        adjectives.append(token.text)
    if token.pos_=="NOUN":
        nouns.append(token.text)
    else:
        others.append(token.text)
for i,(token,word) in enumerate(zip(text,perturbation1)):
    if token.pos_=="ADJ":
        adj = np.random.choice(adjectives)
        while(adj is token.text):
            adj = np.random.choice(adjectives)
        perturbation1[i] = adj
        adjectives.remove(adj)
    if token.pos_=="NOUN":
        noun = np.random.choice(nouns)
        while(noun is token.text and len(nouns)!=1):
            noun = np.random.choice(nouns)
        perturbation1[i] = noun
        nouns.remove(noun)

print("Perturbation1: "+" ".join(perturbation1))

perturbation2 = newtext.copy()
adjectives = []
nouns = []
others = []
for token in text:
    if token.pos_=="ADJ":
        adjectives.append(token.text)
    elif token.pos_=="NOUN":
        nouns.append(token.text)
    else:
        others.append(token.text)
    
for i,(token,word) in enumerate(zip(text,perturbation2)):
    if token.pos_=="ADJ":
        continue
    if token.pos_=="NOUN":
        continue
    else:
        other = np.random.choice(others)
        while(other is token.text and len(others)!=1):
            other = np.random.choice(others)
        perturbation2[i] = other
        others.remove(other)

print("Perturbation2: "+" ".join(perturbation2))

# sampletext = " ".join(newtext.copy())
# phrases = []
# for nc in text.noun_chunks:
#     phrases.append(nc.text)
# for phrase in phrases:
#     sampletext = sampletext.replace(phrase,"")
# sampletext =sampletext.split()
# l = sampletext+phrases
# perturbation3 = np.random.permutation(l)
# print("Perturbation3: "+" ".join(perturbation3))

# sampletext = " ".join(newtext.copy())
# phrases = []
# phrasesrand = []
# for nc in text.noun_chunks:
#     phrases.append(nc.text)
# for nc in text.noun_chunks:
#     phrasesrand.append((np.random.permutation(nc.text.split())).tolist())
# for phrase in phrases:
#     sampletext = sampletext.replace(phrase,"")
# sampletext =sampletext.split()
# print(phrasesrand)
# l = sampletext+phrasesrand
# perturbation4 = np.random.permutation(l)
# print("Perturbation4: "+" ".join(perturbation4))

sampletext = newtext.copy()
numtri = 1+int(len(sampletext)/3)
trigrams = np.random.permutation(np.array([sampletext[3*i:3*(i)+3] for i in range(numtri)],dtype=object)).tolist()
perturbation3 = [item for sublist in trigrams for item in sublist]

print("Perturbation3: "+" ".join(perturbation3))

sampletext = newtext.copy()
numtri = 1+int(len(sampletext)/3)
perturbation4 = []
for i in range(numtri):
    a = np.random.permutation(np.array(sampletext[3*i:(3*i)+3]))
    for word in a:
        perturbation4.append(word)

print("Perturbation4: "+" ".join(perturbation4))


