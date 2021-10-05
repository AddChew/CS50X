# Top2VecApp
Top2VecApp is a desktop application, offering topic modelling capabilities based on [Top2Vec](https://github.com/ddangelov/Top2Vec) algorithm.

## Video demo
Please refer [here]() for a video demo of Top2VecApp in action.

## How the application works
TO DO

## Installation Instructions

#### Run from executable file
1. Download Top2VecApp from [here]() and unzip it

#### Run from source code
1. Clone this repository
```
    $ git clone https://github.com/AddChew/cs50x.git
    $ cd cs50x
```
2. Install the required dependencies
```
    $ pip install -r requirements.txt
```

## Usage Instructions

#### Run from executable file
1. Launch Top2VecApp by double clicking on Top2VecApp.exe
2. Upload CSV file containing the text that you want to cluster and follow the on screen instructions

#### Run from source code
1. Navigate to app folder in cs50x folder
```
    $ cd app
```
2. Launch Top2VecApp
```
    $ python app.py
```
3. Upload CSV file containing the text that you want to cluster and follow the on screen instructions

## Core Technologies
TO DO

## Design choices
The following issues were considered when building the application.

#### Desktop vs web application
Initially, it was conceived for Top2VecApp to be a web application. This is because a web application does not require any prior setup from the user's end (i.e. the user need not install anything to run the application; all he needs is internet access and a web browser). But resource constraints (i.e. insufficient RAM) on Heroku cloud platform made this infeasible.

As such, Top2VecApp pivoted from being a web application to being a desktop application. This is because local machines have significantly more computing power and memory as compared to cloud platforms (i.e. Heroku) and hence can better handle the payload incurred by Top2VecApp. Another reason why Top2VecApp became a desktop application is because user input cannot be trusted. For instance, the user could navigate to urls within the web application via unintended ways or refresh the page even when he is told not to do so, and this would cause the web application to behave unexpectedly. Designing Top2VecApp as a desktop application allows for fine grain control over the application widgets (i.e. what widgets to include and their functionalities). This helps to prevent users from using Top2VecApp in unintended ways. For example, by excluding a url bar widget in Top2VecApp, users are prohibited from navigating to urls in unauthorised ways.

#### Native desktop vs embedded web application
TO DO

#### Packaging of application
Top2VecApp is bundled and distributed as a single executable file which contains all the required dependencies for the application to run. This is to minimise any prior setup required from the user's end. All the user needs to do is to download the executable file and launch it and he is all set to use Top2VecApp.

## Project Navigation
app folder contains the source code for Top2VecApp

#### desktop folder
- gui.py 
    - Contains helper classes for creating the desktop GUI

#### models folder
- encoder.py
    - Contains helper classes for loading the model and running model inference

- pipeline.py
    - Contains helper classes for processing the input text corpus through Top2Vec algorithm and then saving the results to excel files

- tokenizer.py 
    - Contains helper functions and classes to tokenize text

- top2vec.py
    - A lightweight version of the original [Top2Vec](https://github.com/ddangelov/Top2Vec) library

- vocab.txt
    - Contains the tokens used for tokenizing text

- sent-transformer.onnx
    - The natural language processing (NLP) model used for obtaining the text embeddings

#### static folder
- appscripts folder
    - Contains the JavaScript libraries and scripts used in Top2VecApp

- images folder
    - Contains Top2VecApp desktop application icon

- styles folder
    - Contains the stylesheets used for styling Top2VecApp

#### templates folder
- Contains the HTML templates used in Top2VecApp

#### app.py
- Contains the backend logic of Top2VecApp

#### config.py
- Contains the application configurations for Top2VecApp