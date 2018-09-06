#Importar librerias
from config import *
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import time

#Clase  que obtiene los tweets en vivo que comentengan 
#las siguientes palabras : taxibeat, uber_peru, Uber_peru

class StdOutListener(StreamListener):
    def on_data(self, data):
        output = open(r".\data\stream.txt","a")
        output.write("{},{}".format(json.loads(data)['created_at'],json.loads(data)['text']))
        output.close()
        time.sleep(30)
        return True

    def on_error(self, status):
        print("Error")
      
if __name__ == '__main__':

    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    stream.filter(track=['taxibeat','uber_peru','Uber_peru'])