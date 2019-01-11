import DB
import numpy as np
import gensim

topics = DB.aquireDB("mongodb", "topics")
readme_cleaned = DB.aquireDB("mongodb", "readme_cleaned")
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


def next_batch(batch_size=100):
    ids = next_id(batch_size)
    X, Y = data_by_ids(ids)
    return X, Y


def next_id(batch_size=100):
    global global_counter
    global mysql_counter
    global id_list
    if mysql_counter % 100 == 0:
        global_counter += 1
        offset = (global_counter - 1) * batch_size * 100
        cursor.execute("SELECT pid FROM rdLength_sorted2 ORDER BY rdlength ASC LIMIT %s OFFSET %s" % (batch_size * 100, offset))
        ids = cursor.fetchall()
        id_list = []
        for pid in ids:
            id_list.append(int(pid[0]))
        if len(id_list) != batch_size:
            global_counter = 1
            offset = (global_counter - 1) * batch_size * 100
            cursor.execute("SELECT pid FROM rdLength_sorted2 ORDER BY rdlength ASC LIMIT %s OFFSET %s" % (batch_size * 100, offset))
            ids = cursor.fetchall()
            id_list = []
            for pid in ids:
                id_list.append(int(pid[0]))
        mysql_counter = 0
    start_idx = mysql_counter * batch_size
    result = id_list[start_idx:start_idx+batch_size]
    mysql_counter += 1
    return result


def data_by_ids(ids):
    max_length = 0
    temp = readme_cleaned.find({'pid': str(ids[-1])}, {'readme_cleaned': 1, '_id': 0})
    for t in temp:
        max_length = len(t['readme_cleaned'].split(" "))
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
        for xw in xws:
            words = xw["readme_cleaned"].split(" ")
            for word in words:
                try:
                    x_temp.append(model.wv[word])
                except:
                    None
        for i in range(len(x_temp), max_length):
            x_temp.append(np.zeros(200))
        X.append(np.array(x_temp))
    return np.array(X), np.array(Y)