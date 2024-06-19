import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def generate_adversarial_sample(text):
    words = word_tokenize(text)
    adversarial_text = []

    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            # Use the first synonym as a simple example
            adversarial_text.append(synonyms[0])
        else:
            adversarial_text.append(word)

    return ' '.join(adversarial_text)


# Example usage
original_text = "This is a simple example to demonstrate adversarial sample generation."
adversarial_text = generate_adversarial_sample(original_text)

print("Original Text: ", original_text)
print("Adversarial Text: ", adversarial_text)
