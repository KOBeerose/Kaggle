# Bag of Words Meets Bags of Popcorn

<h4>This is my first published notebook on Kaggle About NLP, So I decided ofc to take a look at the Bag of Words Meets Bags of Popcorn Compeition 😄😄<br>
<br>I will be doing a EDA of review texts, some Visualization and Pre-Processing. and finally modelling <br></h4>

**<h2>To get started:</h2>**

Creating and cd to new folder for Dataset
```
cd /D ../input & mkdir word2vec-nlp-tutorial & cd word2vec-nlp-tutorial
```

Downloading Dataset from Kaggle
```
kaggle competitions download -c word2vec-nlp-tutorial
```

Unzipping and deleting .zip file
<h5>For Linux:</h5>

```
unzip word2vec-nlp-tutorial.zip && trash word2vec-nlp-tutorial.zip
```
<h5>For Windows:</h5>

```
PowerShell Expand-Archive -Path "word2vec-nlp-tutorial.zip" -DestinationPath ./ & del /f word2vec-nlp-tutorial.zip
```

Lunching the notebook
```
jupyter notebook "..\..\Meets Bags of Popcorn\meets-bags-of-popcorn-a-beginner-s-notebook.ipynb"
```
