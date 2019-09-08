Hello,


To run the program for generating confHist and evaluate functions, it's necessary to execute the seperate-data program, to put lg files in the appropriate train or test folder. 

If only interested in running segmentation program, it will open inkml filenames in train or test.csv, requiring filepath to the inkml file from the directory of the python files.



To run this program:

***** Train / Test Split *****

- put all training inkml files in 'expressionmatch' folder. Or change path on line 190 in seperate-data.py

- puts .lg GT files from train/test split into appropriate folders, for evaluation tools to work

- set root directory (where python files are located) on line 336 in seperate-data.py

- run seperate-data.py

***** Train Classifier (not necessary) *****

- run train_classifier.py

- produces (compressed) proj2-rf.joblib 

***** Run Segmentation Program *****

- run segmentator.py 

- opens files with extension 'expressmatch\*.inkml'

- Takes about 30 minutes to run test set (2600 files) of provided 8k file training set.

Results are stored in Performance Results folder


