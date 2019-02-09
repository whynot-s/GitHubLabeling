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

    cursor.execute("SELECT max(pid) FROM readme_cleaned_filtered_1954")
    c = cursor.fetchall()
    max_id = 0
    for cc in c:
        max_id = int(cc[0])

    failed = 0
    all_topics = topics.find({}, {'pid': 1, 'topic' : 1, '_id' : 0})
    i = 0
    for topics_list in all_topics:
        i += 1
        if i % 1000 == 0:
            print("Processed %s, Failed %s" % (i, failed))
        pid = topics_list['pid']
        if int(pid) <= max_id:
            continue
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
            readme = readme_cleaned.find({'pid': str(pid)}, {'readme_cleaned': 1, '_id': 0})
            for rd in readme:
                try:
                    rdc = rd['readme_cleaned']
                    cursor.execute(
                        "INSERT INTO readme_cleaned_filtered_1954(pid, labels, rdLength, label_num) VALUES(%s, \'%s\', %s, %s)" % (pid, lids, len(rdc.split(" ")), len(label_ids)))
                    mysql_db.commit()
                    cursor.execute("UPDATE readme_cleaned_filtered_1954 SET readme_cleaned = \'%s\' WHERE pid = %s" % (rdc, pid))
                    mysql_db.commit()
                except Exception:
                    failed += 1
                break
        if i % 1000 == 0:
            print("Processed %s, Failed %s" % (i, failed))


def tokenize():
    import jieba
    import jieba.posseg as pseg
    personalized = ['2d', '3d']
    for word in personalized:
        jieba.add_word(word)
    cursor, mysql_db = DB.aquire_mysql("GitHubLabel")
    i = 1
    while True:
        cursor.execute("SELECT pid, readme_cleaned FROM readme_cleaned_filtered_1954 LIMIT 1000 OFFSET %s" % ((i - 1) * 1000))
        result = cursor.fetchall()
        if len(result) == 0:
            break
        for r in result:
            pid = int(r[0])
            readme = r[1]
            output = ""
            length = 0
            seg_list = pseg.cut(readme)
            for seg in seg_list:
                if seg.word == ' ' or seg.flag == 'm' or (seg.flag == 'x' and seg.word not in personalized):
                    continue
                output += "%s " % seg.word
                length += 1
            cursor.execute("UPDATE readme_cleaned_filtered_1954 SET rc2 = \'%s\', rcLength = %s WHERE pid = %s"
                           % (output, length, pid))
            mysql_db.commit()
        i += 1
        if i % 10 == 0:
            print("Processed %s" % (i * 1000))


def filter_w2v():
    import gensim
    model = gensim.models.Word2Vec.load("/sdpdata2/wjrj/w2v/wiki.model")
    cursor, mysql_db = DB.aquire_mysql("GitHubLabel")
    i = 1
    while True:
        cursor.execute("SELECT pid, rc2 FROM readme_cleaned_filtered_1954 LIMIT 1000 OFFSET %s" % ((i - 1) * 1000))
        result = cursor.fetchall()
        if len(result) == 0:
            break
        for r in result:
            pid = int(r[0])
            readme = r[1]
            output = ""
            insize = 0
            outsize = 0
            for seg in readme.split(" "):
                s = None
                try:
                    s = model.wv[seg]
                except Exception:
                    s = None
                if s is None:
                    outsize += 1
                else:
                    insize += 1
                    output += (seg + " ")
            cursor.execute("UPDATE readme_cleaned_filtered_1954 SET inW2V = %s, outW2V = %s, rc3 = \'%s\' WHERE pid = %s"
                           % (insize, outsize, output, pid))
            mysql_db.commit()
        i += 1
        if i % 10 == 0:
            print("Processed %s" % (i * 1000))


def split_train_and_test_data():
    cursor, mysql_db = DB.aquire_mysql("GitHubLabel")
    for i in range(36860):
        cursor.execute("SELECT pid, rc3, labels FROM readme_cleaned_filtered_1954_train ORDER BY RAND() LIMIT 1")
        result = cursor.fetchall()
        for r in result:
            pid = r[0]
            rc3 = r[1]
            labels = r[2]
            cursor.execute("INSERT INTO readme_cleaned_filtered_1954_test VALUES(%s, \'%s\', \'%s\')" % (pid, labels, rc3))
            cursor.execute("DELETE FROM readme_cleaned_filtered_1954_train WHERE pid = %s" % pid)
            mysql_db.commit()
        if i % 1000 == 0:
            print("Processed %s / 36860" % i)
    print("Done")


split_train_and_test_data()
# tokenize()
# filter_w2v()
