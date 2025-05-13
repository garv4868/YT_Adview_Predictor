# YouTube Ad View Prediction Program

This program predicts YouTube ad views using various machine learning models, including regression techniques and neural networks. It processes video metadata to forecast potential ad view counts.

## Features:
- Cleans and preprocesses YouTube video data (views, likes, dislikes, comments, duration, category)
- Handles outliers and converts non-numeric data (e.g., duration in PT format to seconds)
- Evaluates multiple models: Linear Regression, Support Vector Regressor, Decision Tree, Random Forest, and an Artificial Neural Network
- Generates correlation heatmaps for data analysis
- Selects the best performing model based on RMSE (Root Mean Squared Error)
- Saves predictions to a CSV file

## Usage:
The program takes training data (`train_AdView.csv`) and test data (`test.csv`) as input, processes the features, trains multiple models, and outputs predictions for the test set.

## License:
MIT License

Copyright (c) [2024] [garv4868]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
