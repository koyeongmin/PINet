# key points estimation and point instance segmentation approach for lane detection

- Paper : key points estimation and point instance segmentation approach for lane detection
- Paper Link : https://arxiv.org/abs/2002.06604
- Author : Yeongmin Ko, Jiwon Jun, Donghwuy Ko, Moongu Jeon (Gwanju Institute of Science and Technology)

- This repository is pytorch implement of the above paper. Our poposed method, PINet(Point Intance Network), combines key point estimation and point instance segmentation for lane detection. 

## Dependency
- python ( We tested on python 2.7 )
- pytorch ( We tested on python 1.0.1 with GPU(RTX2080ti))
- opencv
- numpy
- visdom (for visualization)
- sklearn (for evaluation)
- ujon (for evaluation)

## Dataset
This code is developed on tuSimple dataset. You can download the dataset from https://github.com/TuSimple/tusimple-benchmark/issues/3. We recommand to make below structure.

    dataset
      |
      |----train_set/               # training root 
      |------|
      |------|----clips/            # video clips, 3626 clips
      |------|------|
      |------|------|----some_clip/
      |------|------|----...
      |
      |------|----label_data_0313.json      # Label data for lanes
      |------|----label_data_0531.json      # Label data for lanes
      |------|----label_data_0601.json      # Label data for lanes
      |
      |----test_set/               # testing root 
      |------|
      |------|----clips/
      |------|------|
      |------|------|----some_clip/
      |------|------|----...
      |
      |------|----test_label.json           # Test Submission Template
      |------|----test_tasks_0627.json      # Test Submission Template
            
Next, you need to change "train_root_url" and "test_root_url" to your path in "parameters.py".
Finally, you can run "fix_dataset.py", and it will generate dataset according to the number of lanes and save dataset in "datset" directory. (We have uploaded dataset. You can use them.)
            
## Test


## Train
