# Top2VecApp
Top2VecApp is a desktop application, offering topic modelling capabilities based on [Top2Vec](https://github.com/ddangelov/Top2Vec) algorithm.

## Video demo
Please refer [here](https://youtu.be/H4jQL4AT9aM) for a video demo of Top2VecApp in action.

## How the application works
At its core, Top2VecApp is a web application embedded into a desktop application.

When the user launches Top2VecApp, he is directed to the file upload page, where he can upload a CSV file, select the column containing his text corpus and select two other columns (optional) for grouping the text corpus. Each time the file upload page is visited via a GET request (default), the storage folder within the application is cleared of its contents. When the user submits his CSV file as well as his selected columns, the CSV file is temporarily stored in the storage folder within the application.

Upon storing the CSV file, a job is submitted to the worker thread and the user is directed to the progress page. Every second, the main application thread sends a GET request to the worker thread to query about the job status. Once the main application thread receives the signal from the worker thread that the job is completed, the user is directed to the download page where he can download the zipped job results.

Underneath the hood, the job processed by the worker thread consists of the the following stages. 
1.  Once the job is submitted, the CSV file is loaded into memory and its text corpus is tokenized (i.e. individual words are broken up into smaller subwords). 
2.  The tokenized text corpus is then fed through the pretrained transformer-based natural language processing (NLP) model, [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), which outputs the corresponding text embeddings for the input text corpus (i.e. numerical vector representation of text). 
3. Following which, the text embeddings are passed through Uniform Manifold Approximation and Projection [UMAP](https://github.com/lmcinnes/umap) algorithm for dimensionality reduction.
4. Thereafter, the compressed text embeddings are clustered into topics using Hierarchical Density-Based Spatial Clustering of Applications with Noise [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) algorithm.
5. An extractive summary of each topic is obtained based on the top 5 text content closest to the topic centroid in terms of cosine distance.
6. The clustering results and extractive summaries are then written into two separate excel files, zipped together into a single folder for download.  

## Installation Instructions

#### Run from executable file
1. Download Top2VecApp from [here](https://drive.google.com/file/d/1zXvJZV8OW4Tv4CH03LyyNw19qvgLmMyO/view?usp=sharing) and unzip it

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

## Design Considerations
The following issues were considered when building the application.

#### Desktop vs web application
Initially, it was conceived for Top2VecApp to be a web application. This is because a web application does not require any prior setup from the user's end (i.e. the user need not install anything to run the application; all he needs is internet access and a web browser). But resource constraints (i.e. insufficient RAM) on Heroku cloud platform made this infeasible.

As such, Top2VecApp pivoted from being a web application to being a desktop application. This is because local machines have significantly more computing power and memory as compared to cloud platforms (i.e. Heroku) and hence can better handle the payload incurred by Top2VecApp. Another reason why Top2VecApp became a desktop application is because user input cannot be trusted. For instance, the user could navigate to urls within the web application via unintended ways or refresh the page even when he is told not to do so, and this would cause the web application to behave unexpectedly. Designing Top2VecApp as a desktop application allows for fine grain control over the application widgets (i.e. what widgets to include and their functionalities). This helps to prevent users from using Top2VecApp in unintended ways. For example, by excluding a url bar widget in Top2VecApp, users are prohibited from navigating to urls in unauthorised ways.

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