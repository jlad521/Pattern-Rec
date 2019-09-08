Hello,

All necessary files to run the classifier are in this directory. Therefore can just run:

python test-classifier.py

it will generate output csvs for all three test sets (true, true + junk, bonus) 

Additionally, all files can by replicated by running programs in order:

python seperate-data.py
python train-classifier.py
python test-classifier.py


finally, to test results:

python evalSymbols.py true_test_GT.csv kdtree-output-true_test.csv
python evalSymbols.py true_test_GT.csv rf-output-true_test.csv
python evalSymbols.py both_test_GT.csv kdtree-output-both_test.csv
python evalSymbols.py both_test_GT.csv rf-output-both_test.csv

bonus files are included as rf-output-bonus_test.csv and kdtree-output-bonus_test.csv


Note:
File paths are in the form: 

../task2-trainSymb2014(1)/trainingJunk\
../task2-trainSymb2014(1)/trainingSymbols\
../testSymbols/testSymbols\


Best,
Justin & Richard