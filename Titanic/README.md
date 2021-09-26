# Titanic - Machine Learning from Disaster
This is my first published notebook ever, So yeah it can't be about something other than the Titanic Competition 😄

**<h2>To get started:</h2>**

Creating and cd to new folder for Dataset
```
cd /D ../input & mkdir titanic & cd titanic
```

Downloading Dataset from Kaggle
```
kaggle competitions download -c titanic
```

Unzipping and deleting .zip file
<h5>For Linux:</h5>

```
unzip titanic.zip && trash titanic.zip
```
<h5>For Windows:</h5>

```
PowerShell Expand-Archive -Path "titanic.zip" & del /f titanic.zip
```

Lunching the notebook
```
jupyter notebook ..\..\Titanic\getting-started-with-titanic.ipynb
```
