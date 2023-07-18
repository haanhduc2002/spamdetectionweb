#import libraries
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Preparing text data
def handle_message(message): 
  # Step 1: Remove puntuation 
  message1 = [char for char in message if char not in string.punctuation]
  message1 = ''.join(message1)
  # Step 2: Remove stop word
  message2 = [word for word in message1.split() if word.lower() not in stopwords.words('english')]
  message2 = ' '.join(message2)
  # Step 3: Lemmatization
  lemmatizer = WordNetLemmatizer()
  word_list = nltk.word_tokenize(message2)
  result = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
  return result.lower() # Lower casing the processed message