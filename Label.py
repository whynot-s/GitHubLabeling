from pymongo import MongoClient
import pymysql
import json

dbConfig = "";
with open("dbConfig.json", "r") as f:
    dbConfig = json.loads(f.read())

mongoConfig = dbConfig["mongodb"]
client = MongoClient(mongoConfig["host"], mongoConfig["port"])
db = client.github
readme_cleaned = db.readme_cleaned
print("Connected to Mongodb %s:%s %s" % (mongoConfig["host"], mongoConfig["port"], "readme_cleaned"))

mysqlConfig = dbConfig["mysql"]
mysql_db = pymysql.connect(mysqlConfig['host'], mysqlConfig['username'], mysqlConfig['password'], mysqlConfig['db'])
cursor = mysql_db.cursor()
print("Connected to Mysql %s:%s %s" % (mysqlConfig['host'], mysqlConfig['port'], mysqlConfig['db']))

results = readme_cleaned.find({}, {'keyword' : 1}).limit(20)
for result in results:
    print(result)
