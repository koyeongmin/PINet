# key points estimation and point instance segmentation approach for lane detection

- New version is available at https://github.com/koyeongmin/PINet_new
- Python3, pretrained weights of CULane and TuSimple, and higher performance.


## Dependency
- python ( We tested on python 2.7 )
- pytorch ( We tested on pytorch 1.0.1 with GPU(RTX2080ti))
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
            
Next, you need to change "train_root_url" and "test_root_url" to your "train_set" and "test_set" directory path in "parameters.py". For example,

```
# In "parameters.py"
line 54 : train_root_url="<tuSimple_dataset_path>/train_set/"
line 55 : test_root_url="<tuSimple_dataset_path>/test_set/"
```

Finally, you can run "fix_dataset.py", and it will generate dataset according to the number of lanes and save dataset in "dataset" directory. (We have uploaded dataset. You can use them.)
            
## Test
We provide trained model, and it is saved in "savefile" directory. You can run "test.py" for testing, and it has some mode like following functions 
- mode 0 : Visualize results on test set
- mode 1 : Run the model on the given video. If you want to use this mode, enter your video path at line 63 in "test.py"
- mode 2 : Run the model on the given image. If you want to use this mode, enter your image path at line 82 in "test.py"
- mode 3 : Test the model on whole test set, and save result as json file.

You can change mode at line 22 in "parameters.py".

If you want to use other trained model, just change following 2 lines.
```
# In "parameters.py"
line 13 : model_path = "<your model path>/"
# In "test.py"
line 42 : lane_agent.load_weights(<>, "tensor(<>)")
```

If you run "test.py" by mode 3, it generates "test_result.json" file. You can evaluate it by running just "evaluation.py".

Following three lines in "test.py" are for post-processing. If you do not want to use this post-processing, make these three lines comments.
```
# In "test.py"
line 210 : in_x, in_y = eliminate_out(in_x, in_y, confidence, deepcopy(image))
line 211 : in_x, in_y = util.sort_along_y(in_x, in_y)
line 212 : in_x, in_y = eliminate_fewer_points(in_x, in_y)
```
You can get around 96.70% accuracy performance with the post-processing and 96.62% without the post-processing.

## Train
If you want to train from scratch, make line 13 blank in "parameters.py", and run "train.py"
```
# In "parameters.py"
line 13 : model_path = ""
```
"train.py" will save sample result images(in "test_result/"), trained model(in "savefile/"), and evaluation result for some threshold values(0.3, 0.5, 0.7). However, in the most case, around 0.8 show the best performance.

We recommand to make line 210, 211, 212 of "test.py" as comments when you train the model, because post-processing takes more time.

If you want to train from a trained model, just change following 2 lines.
```
# In "parameters.py"
line 13 : model_path = "<your model path>/"
# In "train.py"
line 54 : lane_agent.load_weights(<>, "tensor(<>)")
```
