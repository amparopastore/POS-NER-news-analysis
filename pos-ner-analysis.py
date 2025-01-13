from nltk.tokenize import PunktSentenceTokenizer
from nltk import word_tokenize, pos_tag, ne_chunk

file = open('news article-1.txt', 'r')
news = file.read()

# detect sentences in the given news article 
sentTokenizer = PunktSentenceTokenizer()
news_tokenized = sentTokenizer.tokenize(news)

print("Detected sentences:\n", news_tokenized)

# processing
def process_content():
    try:
        for i, sent in enumerate(news_tokenized):
            print(f"\nSentence {i+1}: {sent}\n") # print sentence, just for readability
            
            words = word_tokenize(sent)
            tagged = pos_tag(words)
            
            # perform Part-of-Speech (POS) on each sentence
            print("Tokens and POS Tags:")
            # compact_tagging = ", ".join([f"({word}/{tag})" for word, tag in tagged])
            print(tagged)
            
            # find name entities including person's name entities and locations 
            named_entities = ne_chunk(tagged)
            print("\nNamed Entities:")
            print(named_entities)  
                  
    except Exception as e:
        print(str(e))
        
process_content()
