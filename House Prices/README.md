# House Prices - Advanced Regression Techniques

<h4>This is my second published notebook on Kaggle, So yeah no wonder it's about the House Prices Competition 😄😄<br><br>I will be doing a simple then advanced EDA, Data Visualization and Pre-Processing. I also will test different approaches and regression techniques to improve my score.<br></h4>

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
PowerShell Expand-Archive -Path "house-prices-advanced-regression-techniques.zip" -DestinationPath ./ & del /f house-prices-advanced-regression-techniques.zip
```

Lunching the notebook
```
jupyter notebook "..\..\House\getting-started-with-house-prices-advanced-regression-techniques.ipynb"
```
