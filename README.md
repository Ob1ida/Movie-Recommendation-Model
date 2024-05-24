# Movie Recommendation system
## NLP Model that recommend you movies based of your text
### [DataSet](https://www.kaggle.com/datasets/ahsanaseer/top-rated-tmdb-movies-10k?fbclid=IwAR2MpWrWpcw2QNCv_FZg2l0sjBh9xAvhrqtnZBO9K-QS6PHI1aHkdB6qLa0)
## Usage
1-open files in vscode

2-make sure to paste your api key or api read access token here


```python

url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer (here)
    }

```

**you can get an api key from the oficial site here**
[the Movie db](https://www.themoviedb.org/)

3-in teminal run the following command
```bash
streamlit run myapp.py
```


## How it works
**TfidVectorizer**
Tfidf works better than CountVectorizer as it also takes the importance of a word into account account.

**tf** : the number of times a word appears in a document divided by totlal numbers of words in that document

**idf**: the logarithm of the number of the number of the documents in the coprus devided by the number of documents where the specific term appears.

<img width="1163" alt="mlp_kan_compare" src="https://github.com/Ob1ida/Movie-Recommendation-Model/assets/96666263/71a9da4e-abd2-44fb-8ce2-9b66ad5b2bd9">




after normlizing the data ,the columns now are combined to a single feature called tags that involves all the movie
details like tags keywords main cast vs.  .

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
now we are ready to fit the model
```python
vectorizer = TfidfVectorizer(stop_words="english")

tfidf = vectorizer.fit_transform(Final_df["tags"])
```
**cosie similarty**
similarity measure refers to distance with dinebsions reoresenting features of the data object in a dataset
Cosine similarity is a mertic, helpful in detemining how similar the data objects are.

to use cosine similarity in python import this:

```python
from sklearn.metrics.pairwise import cosine_similarity
```


## Deployment
**used streamlit library for frontend** 
after running the app all you must do is type any text you want in the textbox to get some similar movies
for example: "mafia"


<img width="400" alt="" src="https://github.com/Ob1ida/Movie-Recommendation-Model/assets/96666263/39528b4f-63d5-4792-b89a-b8025118acb4">

## libraries used

* Pandas  <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="20" height="20"/>
* seaborn  <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="20" height="20"/>
* numpy
* scikit-learn



















# working demo

https://movie-recommendation-model-bizq.onrender.com/
