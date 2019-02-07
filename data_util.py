import DB


def split_and_transfer_data():
    cursor, mysql_db = DB.aquire_mysql("GitHubLabel")
    mdb = DB.aquire_mongo()
    topics = mdb["topics"]
    readme_cleaned = mdb["readme_cleaned"]

    lb_list = {}
    labels = []
    cursor.execute("SELECT label, number FROM Labels_filtered")
    lbs = cursor.fetchall()
    for lb in lbs:
        labels.append(lb[0])
        lb_list[lb[0]] = int(lb[1])
    labels = set(labels)

    failed = 0
    all_topics = topics.find({}, {'pid': 1, 'topic' : 1, '_id' : 0})
    i = 0
    for topics_list in all_topics:
        i += 1
        pid = topics_list['pid']
        flag = False
        label_ids = set([])
        for topic in topics_list['topic']:
            if topic in labels:
                flag = True
                label_ids.add(lb_list[topic])
        if flag:
            lids = ""
            for lid in label_ids:
                lids += "%s;" % lid
            cursor.execute("INSERT INTO readme_cleaned_filtered_1954(pid, labels) VALUES(%s, \'%s\')" % (pid, lids))
            mysql_db.commit()
            readme = readme_cleaned.find({'pid': str(pid)}, {'readme_cleaned': 1, '_id': 0})
            for rd in readme:
                try:
                    cursor.execute("UPDATE readme_cleaned_filtered_1954 SET readme_cleaned = \'%s\' WHERE pid = %s" % (rd['readme_cleaned'], pid))
                    mysql_db.commit()
                except Exception:
                    failed += 1
                break
        if i % 1000 == 0:
            print("Processed %s, Failed %s" % (i, failed))


split_and_transfer_data()