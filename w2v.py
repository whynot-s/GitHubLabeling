import gensim
import DB


def process_readme():
    readme_cleaned = DB.aquireDB("mongodb", "readme_cleaned")
    results = readme_cleaned.find({}, {"readme_cleaned" : 1, "_id" : 0})
    print("Get cursor")
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
    model.wv.save_word2vec_format("/sdpdata2/wjrj/wiki.en.tect.vector_with_readme", binary=False)


if __name__ == "__main__":
    # process_readme()
    train()