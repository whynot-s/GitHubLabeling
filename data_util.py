import DB
import numpy as np


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
    for i in range(31754):
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
            print("Processed %s / 31754" % i)
    print("Done")


def create_vocabulary(word2vec_model_path, name_scope):
    import gensim
    import os
    import pickle
    cache_path = '/sdpdata2/wjrj/GitHubLabeling/cache_vocabulary_label_pik/' + name_scope + "_word_vocabulary.pik"
    print("cache_path:", cache_path, "file_exists:", os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as data_f:
            vocabulary_word2index, vocabulary_index2word = pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index = {}
        vocabulary_index2word = {}
        print("create vocabulary. word2vec_model_path:", word2vec_model_path)
        vocabulary_word2index['PAD'] = 0
        vocabulary_index2word[0] = 'PAD'
        model = gensim.models.Word2Vec.load(word2vec_model_path)
        for i, vocab in enumerate(model.wv.vocab):
            vocabulary_word2index[vocab] = i + 1
            vocabulary_index2word[i + 1] = vocab
        if not os.path.exists(cache_path):
            with open(cache_path, 'a') as data_f:
                pickle.dump((vocabulary_word2index, vocabulary_index2word), data_f)
    return vocabulary_word2index, vocabulary_index2word


def next_batch(offset, batch_size, num_classes, sequence_length, training=True):
    cursor, mysql_db = DB.aquire_mysql("GitHubLabel")
    if training:
        cursor.execute("SELECT rc3, labels FROM readme_cleaned_filtered_1954_train LIMIT %s OFFSET %s"
                       % (batch_size, offset * batch_size))
    else:
        cursor.execute("SELECT rc3, labels FROM readme_cleaned_filtered_1954_test LIMIT %s OFFSET %s"
                       % (batch_size, offset * batch_size))
    results = cursor.fetchall()
    x = []
    y = []
    count = 0
    for r in results:
        rc3 = r[0].split(" ")[:-1]
        labels = [int(v) - 1 for v in r[1].split(";")]
        y.append(np.eye(num_classes)[labels])
        if len(rc3) < sequence_length:
            while len(rc3) < sequence_length:
                rc3.append('PAD')
        else:
            rc3 = rc3[:sequence_length]
        x.append(rc3)
        count += 1
    if count != batch_size:
        return None, None
    return x, y

# split_train_and_test_data()
# tokenize()
# filter_w2v()
