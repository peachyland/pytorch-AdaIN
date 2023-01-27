# MY_CMD="python test.py --content input/content/cornell.jpg --style input/style/woman_with_hat_matisse.jpg"

# MY_CMD="python test.py --content input/content/cornell.jpg --style input/style/flower_of_life.jpg --adv"

MY_CMD="python test.py --content input/content/cornell.jpg --style ./adv/attack3.png"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='2' $MY_CMD
