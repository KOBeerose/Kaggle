# Prediction of Nitrogen 2 Rate

<h4>This is my competition notebook on Kaggle 😁 The objective of the project is to predict the percentage of nitrogen in plants based on satellite data and soil data.The data used are data from 500 cereal plots.<br>
<br>I will be doing a EDA of review texts, some Visualization and Pre-Processing. and finally modelling <br></h4>

**<h2>To get started:</h2>**

Creating and cd to new folder for Dataset
```
cd /D ../input & mkdir prediction-du-taux-dazote-2 & cd prediction-du-taux-dazote-2
```

Downloading Dataset from Kaggle
```
kaggle competitions download -c prediction-du-taux-dazote-2
```

Unzipping and deleting .zip file
<h5>For Linux:</h5>

```
unzip prediction-du-taux-dazote-2.zip && trash prediction-du-taux-dazote-2.zip
```
<h5>For Windows:</h5>

```
PowerShell Expand-Archive -Path "prediction-du-taux-dazote-2.zip" -DestinationPath ./ & del /f prediction-du-taux-dazote-2.zip
```

Lunching the notebook
```
jupyter notebook "..\..\Meets Bags of Popcorn\prediction-of-nitrogen-2-rate.ipynb"
```
