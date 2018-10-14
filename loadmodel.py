import gensim.downloader as api
import spacy
# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8
import numpy as np
import veriservice as vs
import time


# http://opendatacommons.org/licenses/pddl/
model = api.load("glove-wiki-gigaword-100")  # download the model and return as object ready for use
# model.most_similar("car")

# https://dumps.wikimedia.org/legal.html
corpus = api.load('wiki-english-20171001')
nlp = spacy.load('en_core_web_sm')

vc = vs.VeriClient("localhost:10000")
it = 0
error_count = 0
code = 0
for l in corpus:
    # print(l['title'])
    # print(l['section_texts'])
    text = ''.join(l['section_texts'])
    # python -m spacy link en_core_web_md en_default
    doc = nlp(text)
    for sentence in doc.sents:
        words = nlp(sentence.text)
        # print(len(words))
        features = np.array([])
        for token in words:
            try:
                # print( model.wv[token.text.lower()] )
                features = np.append(features, model.wv[token.text.lower().strip()])
                # break
            except:
                error_count+=1
              # word = token.text.lower().strip()
              #if (len(word) > 0):
              # print("Unknown word:_"+token.text.lower().strip()+"_")
        response = vc.insert(features[:150000], l['title'], l['title'], 0)
        code = response.code
        while code == 1 :
            time.sleep(5)
            response = vc.insert(features[:150000], l['title'], l['title'], 0)
            code = response.code
    it+=1
    print(it, code, l['title'])
    if it == 5000 or code == 1:
        break

count = 0
for l in corpus:
    # print(l['title'])
    # print(l['section_texts'])
    text = ''.join(l['section_texts'])
    # python -m spacy link en_core_web_md en_default
    doc = nlp(text)
    for sentence in doc.sents:
        count+=1
    print(count)
