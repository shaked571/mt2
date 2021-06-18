Refael Shaked Greenfeld 305030868
environment - activate conda environment (like part 1):
    >> conda create -n "myenv" python=3.8
    >> conda activate myenv
	>> pip install -r requirements.txt
	#install torch according to your GPU - for example for cuda 10.2
	>> pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html 
	
To run:
Need to have in the root of the files the data files
<root>\data\{train, dev, test}.{src, trg}

trainig:
>> python part1_train.py

prediction (after you ran train and the script dump a model):

>> python part1_evaluation.py
