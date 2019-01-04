from pymongo import MongoClient
import pymysql
import json

dbConfig = "";
with open("dbConfig.json", "r") as f:
    dbConfig = json.loads(f.read())

mongoConfig = dbConfig["mongodb"]
client = MongoClient(mongoConfig["host"], mongoConfig["port"])
db = client.github
topics = db.topics
print("Connected to Mongodb %s:%s %s" % (mongoConfig["host"], mongoConfig["port"], "topics"))

mysqlConfig = dbConfig["mysql"]
mysql_db = pymysql.connect(mysqlConfig['host'], mysqlConfig['username'], mysqlConfig['password'], mysqlConfig['db'])
cursor = mysql_db.cursor()
print("Connected to Mysql %s:%s %s" % (mysqlConfig['host'], mysqlConfig['port'], mysqlConfig['db']))

results = topics.find({}, {'topic' : 1, '_id' : 0})
numbers = {}

i = 0
for result in results:
    i += 1
    topics = result['topic']
    for topic in topics:
        if topic in numbers:
            numbers[topic] += 1
        else:
            numbers[topic] = 1
    if i % 1000 == 0:
        print("PROCESS: %s" % i)

j = 0
for key, value in numbers.items():
    j += 1
    cursor.execute("INSERT INTO Labels VALUES(\"%s\", %s)" % (key, value))
    if j % 1000 == 0:
        print("INSERT: %s/%s" % (j, i))

mysql_db.commit()

