# This shell scripts triggers the python code with the required parameter and reduces effort by not triggering
# the model one by one
#!/bin/sh
python src/train.py --fold 0 --model rf
