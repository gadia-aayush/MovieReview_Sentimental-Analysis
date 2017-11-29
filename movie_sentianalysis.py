from nltk.classify import NaiveBayesClassifier

def word_feats(words):
    return dict([(word, True) for word in words])
    
positive_vocab = ['awesome','outstanding','fantastic','terrific','good','nice','great']
negative_vocab = ['bad','terrible','useless','hate']

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]

train_set = negative_features + positive_features
classifier = NaiveBayesClassifier.train(train_set)

neg = 0
pos = 0
sentence = "Awesome movie, I liked it. Simply terrific"
words = sentence.lower().split(' ')
for word in words:
    classResult = classifier.classify( word_feats(word))
if classResult == 'neg':
    neg = neg + 1
if classResult == 'pos':
    pos = pos + 1
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))
print('Overall Sentiment is : ' , (float(pos)/len(words)) - (float(neg)/len(words)))

