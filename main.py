import spacy
nlp = spacy.load('en_core_web_sm')
txt = 'Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.'
doc = nlp(txt)

print([ent.text for ent in doc.ents])
print([(t.ent_type_, t.ent_iob_) for t in doc])
