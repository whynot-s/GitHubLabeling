import gensim
import DB


def process_readme():
    readme_cleaned = DB.aquireDB("mongodb", "readme_cleaned")
    results = readme_cleaned.find({}, {"readme_cleaned" : 1, "_id" : 0})
    i = 0
    with open("/sdpdata2/wjrj/github.en.text", "w") as f:
        for result in results:
            i += 1
            f.write("%s\n" % result['readme_cleaned'])
            if i % 1000 == 0:
                print("Processed %d" % i)
        f.close()


def train():
    model = gensim.models.Word2Vec.load("/sdpdata2/wjrj/wiki.en.text.model")
    model.train(gensim.models.word2vec.LineSentence("/sdpdata2/wjrj/github.en.text"),
                total_examples=model.corpus_count, epochs=model.iter)
    model.save("/sdpdata2/wjrj/wiki.en.text.model_with_readme")
    model.wv.save_word2vec_format("/sdpdata2/wjrj/wiki.en.text.vector_with_readme", binary=False)


def create_bucket():
    cursor, mysql_db = DB.aquireDB("mysql", "GitHubLabel")
    readme_cleaned = DB.aquireDB("mongodb", "readme_cleaned")
    results = readme_cleaned.find({}, {"readme_cleaned": 1, "_id": 0, "pid": 1})
    i = 0
    data = []
    for result in results:
        i += 1
        data.append(tuple([int(result["pid"]), result["readme_cleaned"], len(result["readme_cleaned"].split(" "))]))
        if i % 1000 == 0:
            print("Processed %d" % i)
            cursor.executemany("INSERT INTO rdLength VALUES(%s, %s, %s)", data)
            mysql_db.commit()
            data = []
    if len(data) != 0:
        cursor.executemany("INSERT INTO rdLength VALUES(%s, %s, %s)", data)
        mysql_db.commit()


if __name__ == "__main__":
    # process_readme()
    # train()
    create_bucket()