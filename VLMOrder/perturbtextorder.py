import spacy
import numpy as np
import re
nlp = spacy.load("en_core_web_sm")
# text = nlp("remarkable scene with a blue ball behind a green chair")
# newtext = "remarkable scene with a blue ball behind a green chair".split()
# print("Original Text: "+" ".join(newtext))


# Evaluation Perturbations
def nounadjshuf(doc):
    text = nlp(doc)
    newtext = doc.split()
    
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

    return " ".join(perturbation1)

def nonnounadjshuf(doc):
    text = nlp(doc)
    newtext = doc.split()

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

    return " ".join(perturbation2)

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

def trigramshuf(doc):
    text = nlp(doc)
    newtext = doc.split()
    sampletext = newtext.copy()
    numtri = 1+int(len(sampletext)/3)
    trigrams = np.random.permutation(np.array([sampletext[3*i:3*(i)+3] for i in range(numtri)],dtype=object)).tolist()
    perturbation3 = [item for sublist in trigrams for item in sublist]

    return " ".join(perturbation3)

def wordtrigramshuf(doc):
    text = nlp(doc)
    newtext = doc.split()
    sampletext = newtext.copy()
    numtri = 1+int(len(sampletext)/3)
    perturbation4 = []
    for i in range(numtri):
        a = np.random.permutation(np.array(sampletext[3*i:(3*i)+3]))
        for word in a:
            perturbation4.append(word)

    return " ".join(perturbation4)


# NegCLIP Hard Negatives

# text = nlp("The old man is slowly eating the sandwich and the pretty woman is calmly watching the television ")
# newtext = "The old man is slowly eating the sandwich and the pretty woman is calmly watching the television".split()
# print("Original Statement: "+" ".join(newtext))
def get2random(ls):
    if(len(ls)==2):
        return np.flip(ls)
    
    two =  np.random.choice(ls,2,False)
    if(ls.index(two[0])<ls.index(two[1])):
        return np.flip(two)
    return two

# Swapping Noun Phrases:
def nounshuf(doc):
    text = nlp(doc)
    newtext = doc.split()
    adjectives = []
    nouns = []
    adverbs = []
    for token in text:
        if token.pos_=="ADJ":
            adjectives.append(token.text)
        elif token.pos_=="NOUN":
            nouns.append(token.text)
        elif token.pos_=="ADV":
            adverbs.append(token.text)
    if(len(nouns)>1):
        word1,word2 = get2random(nouns)
        index1, index2 = newtext.index(word1), newtext.index(word2)
        nounswap = newtext.copy()
        nounswap[index1],nounswap[index2] = newtext[index2],newtext[index1]
        return " ".join(nounswap)
    else:
        return doc

def advshuf(doc):
    text = nlp(doc)
    newtext = doc.split()
    adjectives = []
    nouns = []
    adverbs = []
    for token in text:
        if token.pos_=="ADJ":
            adjectives.append(token.text)
        elif token.pos_=="NOUN":
            nouns.append(token.text)
        elif token.pos_=="ADV":
            adverbs.append(token.text)
    if(len(adverbs)>1):
        word1,word2 = get2random(adverbs)
        index1, index2 = newtext.index(word1), newtext.index(word2)
        advswap = newtext.copy()
        advswap[index1],advswap[index2] = newtext[index2],newtext[index1]
        return " ".join(advswap)
    else:
        return doc

def adjshuf(doc):
    text = nlp(doc)
    newtext = doc.split()
    adjectives = []
    nouns = []
    adverbs = []
    for token in text:
        if token.pos_=="ADJ":
            adjectives.append(token.text)
        elif token.pos_=="NOUN":
            nouns.append(token.text)
        elif token.pos_=="ADV":
            adverbs.append(token.text)

    if(len(adjectives)>1):
        word1,word2 = get2random(adjectives)
        index1, index2 = newtext.index(word1), newtext.index(word2)
        adjswap = newtext.copy()
        adjswap[index1],adjswap[index2] = newtext[index2],newtext[index1]
        return " ".join(adjswap)

def nounchunkshuf(doc):
    text = nlp(doc)
    newtext = doc.split()
    NCs = []
    for nc in text.noun_chunks:
        NCs.append(nc.text)
    if(len(NCs)>1):
        phrase1,phrase2 = get2random(NCs)
        ncswap = " ".join(newtext.copy())
        ncswap = ncswap.replace(phrase1,"----REPLACE @ PHRASE @ ONE----")
        ncswap = ncswap.replace(phrase2,"----REPLACE @ PHRASE @ TWO----")
        ncswap = ncswap.replace("----REPLACE @ PHRASE @ ONE----",phrase2)
        ncswap = ncswap.replace("----REPLACE @ PHRASE @ TWO----",phrase1)
        return ncswap


# text1 = "remarkable scene with a blue ball behind a green chair"
# text2 = "The old man is slowly eating the sandwich and the pretty woman is calmly watching the television"
# print("Original Text 1",text1)
# print(nounadjshuf(text1))
# print(nonnounadjshuf(text1))
# print(trigramshuf(text1))
# print(wordtrigramshuf(text1))
# print("Original Text 2",text2)
# print(nounshuf(text2))
# print(adjshuf(text2))
# print(advshuf(text2))
# print(nounchunkshuf(text2))