import DB
import numpy as np
import gensim

cursor, mysql_db = DB.aquireDB("mysql", "GitHubLabel")
model = gensim.models.Word2Vec.load("/sdpdata2/wjrj/wiki.en.text.model_with_readme")
print("Word2Vec Model Loaded")
global_counter = 0
mysql_counter = 0
id_list = []
lb_list = {}

cursor.execute("SELECT label, number FROM Labels_filtered")
lbs = cursor.fetchall()
for lb in lbs:
    lb_list[lb[0]] = int(lb[1])


def next_batch(sequence_length, batch_size=100):
    ids = next_id(batch_size)
    X, Y = data_by_ids(ids, sequence_length)
    print(np.shape(X), np.shape(Y))
    return X, Y


def next_id(batch_size=100):
    global global_counter
    global mysql_counter
    global id_list
    if mysql_counter % 100 == 0:
        global_counter += 1
        offset = (global_counter - 1) * batch_size * 100
        cursor.execute("SELECT pid FROM rdLength_sorted2 LIMIT %s OFFSET %s" % (batch_size * 100, offset))
        ids = cursor.fetchall()
        id_list = []
        for pid in ids:
            id_list.append(int(pid[0]))
        if len(id_list) != batch_size:
            global_counter = 1
            offset = (global_counter - 1) * batch_size * 100
            cursor.execute("SELECT pid FROM rdLength_sorted2 LIMIT %s OFFSET %s" % (batch_size * 100, offset))
            ids = cursor.fetchall()
            id_list = []
            for pid in ids:
                id_list.append(int(pid[0]))
        mysql_counter = 0
    start_idx = mysql_counter * batch_size
    result = id_list[start_idx:start_idx+batch_size]
    mysql_counter += 1
    return result


def data_by_ids(ids, sequence_length):
    topics = DB.aquireDB("mongodb", "topics")
    readme_cleaned = DB.aquireDB("mongodb", "readme_cleaned")
    max_length = sequence_length
    X = []
    Y = []
    for pid in ids:
        x_temp = []
        y_temp = np.zeros(len(lb_list))
        ys = topics.find({'pid': str(pid)}, {'topic': 1, '_id': 0})
        for y in ys:
            for topic in y['topic']:
                if topic in lb_list:
                    y_temp[lb_list[topic] - 1] = 1
        Y.append(np.array(y_temp))
        xws = readme_cleaned.find({'pid': str(pid)}, {'readme_cleaned': 1, '_id': 0})
        count = 0
        flag = False
        for xw in xws:
            words = xw["readme_cleaned"].split(" ")
            for word in words:
                try:
                    x_temp.append(model.wv[word.strip()])
                    count += 1
                    if count == max_length:
                        flag = True
                        break
                except:
                    None
            if flag:
                break
        for i in range(len(x_temp), max_length):
            x_temp.append(np.zeros(200))
        X.append(np.array(x_temp))
    return np.array(X), np.array(Y)


def mock(batch_size, size, embedding_size, classes_num):
    return np.random.random_sample((batch_size, size, embedding_size)), np.eye(classes_num)[np.random.randint(classes_num, size=batch_size)]