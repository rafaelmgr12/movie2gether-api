import unittest
import os 
import pymongo
from dotenv import load_dotenv

load_dotenv()

password = os.getenv('PASSWORD')

url = "mongodb+srv://rafael:"+password+"@cluster0.uabkvrs.mongodb.net/?retryWrites=true&w=majority" 


class TestMongoConnections(unittest.TestCase):

    def test_connection(self):
        client = pymongo.MongoClient(url)
        get_database = client.get_database('test')
        self.assertEqual(get_database.name, 'test')
 
    def test_create_a_post(self):
        client = pymongo.MongoClient(url)
        get_database = client.get_database('test')
        collection = get_database.get_collection('posts')
        post = {'_id':'1','title': 'Python and MongoDB', 'content': 'PyMongo is fun, you guys', 'author': 'Scott '}
        result = collection.insert_one(post)
        self.assertEqual(result.inserted_id, post['_id'])

if __name__ == '__main__':
    unittest.main()