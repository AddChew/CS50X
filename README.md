# Top2VecApp
Top2VecApp is a desktop application, offering topic modelling capabilities based on [Top2Vec](https://github.com/ddangelov/Top2Vec) algorithm.

## Video demo
Please refer [here]() for a video demo of Top2VecApp in action.

## How the application works
placeholder

## Installation Instructions

## Usage Instructions

## Design choices
The following issues were considered when building the application.

#### Desktop vs web application
Initially, it was conceived for Top2VecApp to be a web application. This is because a web application does not require any prior setup from the user's end (i.e. the user need not install anything to run the application; all he needs is internet access and a web browser). But resource constraints (i.e. insufficient RAM) on Heroku cloud platform made this infeasible.

As such, Top2VecApp pivoted from being a web application to being a desktop application. This is because local machines have significantly more computing power and memory as compared to cloud platforms (i.e. Heroku) and hence can better handle the payload incurred by Top2VecApp. Another reason why Top2VecApp became a desktop application is because user input cannot be trusted. For instance, the user could navigate to urls within the web application via unintended ways or refresh the page even when he is told not to do so, and this would cause the web application to behave unexpectedly. Designing Top2VecApp as a desktop application allows for fine grain control over the application widgets (i.e. what widgets to include and their functionalities). This helps to prevent users from using Top2VecApp in unintended ways. For example, by excluding a url bar widget in Top2VecApp, users are prohibited from navigating to urls in unauthorised ways.

#### Native desktop vs embedded web application


#### Packaging of application
Top2VecApp is bundled and distributed as a single executable file which contains all the required dependencies for the application to run. This is to minimise any prior setup required from the user's end. All the user needs to do is to download the executable file and launch it and he is all set to use Top2VecApp.

## Installation Instructions (If you want to run from the executable file)
This method does not require any prior setup (i.e. you do not need to have Python or Anaconda installed on your computer).
1. Download the zipped folder containing the executable file from [here](https://drive.google.com/file/d/1yU9BUdH2x0CBRcIfkNxnclKp7Jz4id3Z/view?usp=sharing).
2. Unzip the zipped folder.
3. Launch Top2VecApp.exe from within the unzipped folder.
4. Test Top2VecApp with the provided test files (You can try with your own files also). The test files can be found [here](https://drive.google.com/drive/folders/1JoZ1MN-rBxCfxl1WuXAt6kCZaWl7iccx?usp=sharing).

## Installation Instructions (If you want to run from the Python source code)
This set of instructions assumes that you already have anaconda prompt installed on your computer.
#### For first-time users:
1. Download the zipped folder containing the source code from offline branch (Click on Code and then Download ZIP).
2. Unzip the zipped folder.
3. Launch anaconda prompt.
4. Navigate to where the unzipped folder is. For example, if your unzipped folder is in Downloads directory, then run the following commands in anaconda prompt:
```
  cd downloads
  cd Top2VecApp-offline
```
5. Create a new Python environment by running the following commands in anaconda prompt:
```
  conda create -n top2vec python=3.7
  conda activate top2vec
```
6. Install the required dependencies by running the following command in anaconda prompt (This might take a while):
```
  pip install -r requirements.txt
```
7. Run Top2VecApp by running the following command in anaconda prompt:
```
  python application.py
```
8. Test Top2VecApp with the provided test files (You can try with your own files also). The test files can be found [here](https://drive.google.com/drive/folders/1JoZ1MN-rBxCfxl1WuXAt6kCZaWl7iccx?usp=sharing).

#### For returning users:
1. Download the zipped folder containing the source code from offline branch (Click on Code and then Download ZIP). This is to ensure that you have the most updated version of the application.
2. Unzip the zipped folder.
3. Launch anaconda prompt.
4. Navigate to where the unzipped folder is. For example, if your unzipped folder is in Downloads directory, then run the following commands in anaconda prompt:
```
  cd downloads
  cd Top2VecApp-offline
```
5. Activate your python environment for Top2VecApp by running the following command in anaconda prompt:
```
  conda activate top2vec
```
6. Run Top2VecApp by running the following command in anaconda prompt:
```
  python application.py
```
7. Test Top2VecApp with the provided test files (You can try with your own files also). The test files can be found [here](https://drive.google.com/drive/folders/1JoZ1MN-rBxCfxl1WuXAt6kCZaWl7iccx?usp=sharing).

## Packaging Instructions
1. Launch anaconda prompt.
2. Navigate to where the unzipped Top2VecApp-offline folder is.
3. Install pyinstaller if you have not done so, by running the following command in anaconda prompt:
```
  pip install pyinstaller
```
6. Run the following command in anaconda prompt to package the application into a single executable file for distribution
```
pyinstaller -w -F --add-data "templates;templates" --add-data "static;static" --add-data "models;models" --add-data "storage;storage" --add-data "desktop;desktop" --icon appicon.ico application.py
```
