import os,re
import nltk
import inflect
p = inflect.engine()
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))



def text_preprocesser(text):
    # remove all non-alphabet characters
    modified_text = re.sub(r'[^a-zA-Z]', ' ', text)
    # remove all non-ascii characters
    modified_text = "".join(ch for ch in modified_text if ord(ch) < 128)
    # convert to lowercase
    modified_text = modified_text.lower() 
    tokens =[word for sent in nltk.sent_tokenize(modified_text) for word in nltk.word_tokenize(sent)]
    allowed_tokens = []
    for token in tokens:
        # words must contain 2 letters
        if re.search('[a-zA-Z]{2,}', token):
            allowed_tokens.append(token)     
    tokens = [t for t in allowed_tokens]
    return tokens


def clean_word(word):
	word = text_preprocesser(word)[0]
	word =word.lower()
	word = word.replace('+','').replace('bacon', 'beef').replace("{", '').replace("}", "").replace("'", '').replace('+', '').replace('_', '').replace('-', '')
	word = word.replace("(", '').replace(")", "").replace("=", '').replace('*', '')
	word =word.replace("patti","").replace("vdrj67a", "").replace('[',"").replace("]", "").replace("'", "").replace('(',"").replace(")", "")  

	return word

### define Superingredients:
def get_ing(file):
	with open(file, 'r') as f:
		fr = []
		for line in f:
			line = nltk.word_tokenize(line.replace("patti","").replace("vdrj67a", "").replace('[',"").replace("]", "").replace("'", "").replace('(',"").replace(")", "")  )
			for frt in line:
				if frt !=",":
					frt = frt.lower()
					if p.singular_noun(frt)==False: fr.append(p.plural_noun(frt))
					else: fr.append(p.singular_noun(frt))
					fr.append(frt)
	return fr
superingredients = {}
covered={}
# superingredients['dairy'] = ['buttermilk', 'margarine', 'butter', 'butteroil' 'cheese', 'cottage', 'milk', 'ricotta', 'sour', 'cream', 'yogurt', 'flavored', 'plain']
superingredients ['fruits'] = get_ing('superingredients/fruits.txt')
superingredients ['grains'] = get_ing('superingredients/grains.txt')
superingredients ['sides'] = get_ing('superingredients/sides.txt')
superingredients ['proteins'] = get_ing('superingredients/proteins.txt')
superingredients ['seasonings'] = get_ing('superingredients/seasonings.txt')
superingredients ['vegetables'] = get_ing('superingredients/vegetables.txt')
superingredients ['drinks'] = get_ing('superingredients/drinks.txt')
superingredients ['dairy'] = get_ing('superingredients/dairy.txt')


def vocab_builder(file_name):
    with open(file_name, 'r') as rf:
        for line in rf:
            for word in nltk.word_tokenize(line):
                covered[word] = 0

def list_to_string(l):
    st = l[0]
    for t in l[1:-1]:
        st = st+" "+t
    return st+" "+l[-1]

def type_gen(file_name):
    with open(file_name, 'r') as rf, open('../recipe_type/'+file_name, 'wb') as wf:
        for line in rf:
            ml = [ covered[word] if covered[word]!=0 and word not in stopWords else word for word in nltk.word_tokenize(line)]
            if len(ml)!=0: ml = list_to_string(ml)
            else:ml =""
            wf.write(ml+"\n")
                


file_name = 'valid.txt'
vocab_builder(file_name)
file_name = 'train.txt'
vocab_builder(file_name)
file_name = 'test.txt'
vocab_builder(file_name)
print len(covered)


#### MUST BE AFTER VOCAB BUILDER
for si, iss in superingredients.items():
    for i in iss:
        # if i =='potatoes': print 'found', si
        covered[i] = si
########

file_name = 'valid.txt'
type_gen(file_name)
file_name = 'train.txt'
type_gen(file_name)
file_name = 'test.txt'
type_gen(file_name)


