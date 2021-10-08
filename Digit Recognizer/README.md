# Digit Recognizer - Learn computer vision fundamentals with the famous MNIST data
<h4>This is my third published notebook on Kaggle. Well! Well you guess it! it's gonna be about the Digit Recognizer Competition 😄😄<br><br>I will be doing a simple EDA and Pre-Processing, the I will Build different models<br><br></h4>

**<h2>To get started:</h2>**

Creating and cd to new folder for Dataset
```
cd /D ../input & mkdir digit-recognizer & cd digit-recognizer
```

Downloading Dataset from Kaggle
```
kaggle competitions download -c digit-recognizer
```

Unzipping and deleting .zip file
<h5>For Linux:</h5>

```
unzip digit-recognizer.zip && trash digit-recognizer.zip
```
<h5>For Windows:</h5>

```
PowerShell Expand-Archive -Path "digit-recognizer.zip" -DestinationPath ./ & del /f digit-recognizer.zip
```

Lunching the notebook
```
jupyter notebook "..\..\Digit Recognizer\digit-recognizer-a-beginner-s-notebook.ipynb"
```
