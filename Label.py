import DB


def filter_labels():
    topics = DB.aquireDB("mongodb", "topics")
    cursor, mysql_db = DB.aquireDB("mysql", "GitHubLabel")
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


def add_no():
    cursor, mysql_db = DB.aquireDB("mysql", "GitHubLabel")
    i = 0
    cursor.execute("SELECT label FROM Labels_filtered")
    results = cursor.fetchall()
    for result in results:
        i += 1
        cursor.execute("UPDATE Labels_filtered SET number = %s WHERE label = \"%s\"" % (i, result[0]))
    mysql_db.commit()

if __name__ == "__main__":
    filter_labels()
    add_no()