import nltk
import csv
from bs4 import BeautifulSoup


def read_data(file_data):
   """
   Lectura de archivo .csv que retorna una lista de tweets y labels.

   Parametros:
   file_data -- ruta 

   Excepciones:

   
   """
   labels = []
   tweets = []

   with open(file_data,'r') as f:
      for line in f:
         stream_line = line.split('\t')
         labels.append(stream_line[0].decode('iso-8859-1').encode('utf8'))
         tweets.append(stream_line[1].decode('iso-8859-1').encode('utf8'))

   tweets = clean_data(tweets)
   return tweets, labels

def clean_data(data):
   clean_uni_data = []
   for text in data:
      text = BeautifulSoup(text, 'html.parser').getText()
      #strips html formatting and converts to unicode
      clean_uni_data.append(text)
   return clean_uni_data










