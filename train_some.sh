python experiment3.py --train=True --dataset='mnist' --mode='greedy' --model_folder='models_median/mnist/greedy/jan30/' -k=5 -T=3 -S=30
python experiment3.py --train=True --dataset='mnist' --mode='vanilla' --model_folder='models_median/mnist/vanilla/jan30/' -k=5 -T=3 -S=30
python experiment3.py --train=True --dataset='mnist' --mode='random' --model_folder='models_median/mnist/random/jan30/' -k=5 -T=3 -S=30

python experiment3.py --train=True --dataset='mnist' --mode='vanilla' --model_folder='models_median/mnist/vanilla/jan30/' -k=10 -T=3 -S=30
python experiment3.py --train=True --dataset='mnist' --mode='greedy' --model_folder='models_median/mnist/greedy/jan30/' -k=10 -T=3 -S=30
python experiment3.py --train=True --dataset='mnist' --mode='random' --model_folder='models_median/mnist/random/jan30/' -k=10 -T=3 -S=30

python experiment3.py --train=True --dataset='mnist' --mode='vanilla' --model_folder='models_median/mnist/vanilla/jan30/' -k=20 -T=3 -S=30
python experiment3.py --train=True --dataset='mnist' --mode='greedy' --model_folder='models_median/mnist/greedy/jan30/' -k=20 -T=3 -S=30
python experiment3.py --train=True --dataset='mnist' --mode='random' --model_folder='models_median/mnist/random/jan30/' -k=20 -T=3 -S=30
