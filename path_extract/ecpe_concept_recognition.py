import configparser
import json
import spacy
from spacy.matcher import Matcher
import sys
import timeit
from tqdm import tqdm
import numpy as np
blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])


concept_vocab = set()
config = configparser.ConfigParser()
config.read(".../grounding/paths.cfg")
with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]

def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_"," "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc])) # all lemma
    return lcs

def load_matcher(nlp):
    config = configparser.ConfigParser()
    config.read(".../grounding/paths.cfg")
    with open(config["paths"]["matcher_patterns"], "r", encoding="utf8") as f:
        all_patterns = json.load(f)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in tqdm(all_patterns.items(), desc="Adding patterns to Matcher."):
        matcher.add(concept, None, pattern)
    return matcher

def ground_mentioned_concepts(nlp, matcher, s, ans = ""):
    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    for match_id, start, end in matches:

        span = doc[start:end].text  # the matched span

        if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
            continue
        original_concept = nlp.vocab.strings[match_id]
        # print("Matched '" + span + "' to the rule '" + string_id)


        if len(original_concept.split("_")) == 1:
            original_concept = list(lemmatize(nlp, original_concept))[0]

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].add(original_concept)


    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)

        concepts_sorted.sort(key=len)


        shortest = concepts_sorted[0:3] #
        for c in shortest:
            if c in blacklist:
                continue
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect)>0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)


    return mentioned_concepts

def hard_ground(nlp, sent):
    global cpnet_vocab
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = "_".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    return res

def match_mentioned_concepts(nlp, sents ):
    matcher = load_matcher(nlp)


    question_concepts = ground_mentioned_concepts(nlp, matcher, sents)
    if len(question_concepts)==0:

        question_concepts = hard_ground(nlp, sents) # not very possible

    
    return list(question_concepts)

def process(filename):


    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    with open(filename, 'r') as fcc_file:
        lines = json.load(fcc_file)

    document=[]
    for line in tqdm(lines, desc="loading file"):
        doc_one={}
        doc_one['doc_id']=line['doc_id'] 
        doc_one['doc_len']=line['doc_len'] 
        doc_one['pairs']=line['pairs']
        clauses=[]
        for clause_i in line['clauses']:
            clause_one={}
            clause_one['clause_id']=clause_i['clause_id']
            res = match_mentioned_concepts(nlp, sents=clause_i['clause'])
            clause_one['clause_concept_recognition']=res
            clauses.append(clause_one)
        doc_one['clauses'] =clauses
        document.append(doc_one)


        


    with open('.../all_data_pair_concept.json', 'w') as fo:
        json.dump(document, fo)



def test():
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    #nlp.add_pipe('sentencizer')
    res = match_mentioned_concepts(nlp, sents=["Sometimes people say that someone stupid has no swimming pool."], answers=["swimming pool"])
    print(res)

# "sent": "Watch television do children require to grow up healthy.", "ans": "watch television",
if __name__ == "__main__":
    #process(sys.argv[1], int(sys.argv[2]))
    process('.../all_data_pair.json')

