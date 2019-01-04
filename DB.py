from pymongo import MongoClient
import pymysql
import json

def aquireDB(dbType, dbname):
    with open("dbConfig.json", "r") as f:
        dbConfig = json.loads(f.read())
    if dbType == "mongodb":
        mongoConfig = dbConfig["mongodb"]
        client = MongoClient(mongoConfig["host"], mongoConfig["port"])
        db = client.github
        collection = db[dbname]
        print("Connected to Mongodb %s:%s %s" % (mongoConfig["host"], mongoConfig["port"], "topics"))
        return collection
    elif dbType == "mysql":
        mysqlConfig = dbConfig["mysql"]
        mysql_db = pymysql.connect(mysqlConfig['host'], mysqlConfig['username'], mysqlConfig['password'], dbname)
        cursor = mysql_db.cursor()
        print("Connected to Mysql %s:%s %s" % (mysqlConfig['host'], mysqlConfig['port'], dbname))
        return cursor, mysql_db
    else:
        return None