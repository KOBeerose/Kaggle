# Titanic - Machine Learning from Disaster
This is my first published notebook ever, So yeah it can't be about something other than the Titanic Competition 😄

**<h2>To get started:</h2>**

Creating and cd to new folder for Dataset
```
cd /D ../input & mkdir house-prices-advanced-regression-techniques & cd house-prices-advanced-regression-techniques
```

Downloading Dataset from Kaggle
```
kaggle competitions download -c house-prices-advanced-regression-techniques
```

Unzipping and deleting .zip file
<h5>For Linux:</h5>

```
unzip house-prices-advanced-regression-techniques.zip && trash house-prices-advanced-regression-techniques.zip
```
<h5>For Windows:</h5>

```
PowerShell Expand-Archive -Path "house-prices-advanced-regression-techniques.zip" & del /f house-prices-advanced-regression-techniques.zip
```

Lunching the notebook
```
jupyter notebook ..\..\House\getting-started-with-house-prices-advanced-regression-techniques.ipynb
```
