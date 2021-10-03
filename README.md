# Top2VecApp
Desktop Application for Top2Vec

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
