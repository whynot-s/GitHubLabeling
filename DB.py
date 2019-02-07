from pymongo import MongoClient
import pymysql
import json

def aquire_mysql(dbname):
    with open("dbConfig.json", "r") as f:
        dbConfig = json.loads(f.read())
        mysqlConfig = dbConfig["mysql"]
        mysql_db = pymysql.connect(mysqlConfig['host'], mysqlConfig['username'], mysqlConfig['password'], dbname)
        cursor = mysql_db.cursor()
        print("Connected to Mysql %s:%s %s" % (mysqlConfig['host'], mysqlConfig['port'], dbname))
        return cursor, mysql_db

def aquire_mongo():
    with open("dbConfig.json", "r") as f:
        dbConfig = json.loads(f.read())
        mongoConfig = dbConfig["mongodb"]
        client = MongoClient(mongoConfig["host"], mongoConfig["port"])
        db = client.github
        print("Connected to Mongodb %s:%s" % (mongoConfig["host"], mongoConfig["port"]))
        return db
