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

results = readme_cleaned.find({}, {'keyword' : 1, '_id' : 0})
numbers = {}

i = 0
for result in results:
    i += 1
    keywords = result['keyword']
    for keyword in keywords:
        temp = numbers[keyword]
        if not temp:
            numbers[keyword] = 1
        else:
            numbers[keyword] = temp + 1
    if i % 1000 == 0:
        print("PROCESS: %s", i)

j = 0
for key, value in numbers.items():
    j += 1
    cursor.execute("INSERT INTO Labels VALUES(\"%s\", %s)" % (key, value))
    if j % 1000 == 0:
        print("INSERT: %s/%s", j, i)

mysql_db.commit()

