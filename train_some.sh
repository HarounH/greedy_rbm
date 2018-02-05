
python experiment.py --train=True --dataset='caltech' --mode='vanilla' --model_folder='models/caltech/vanilla/jan30/' -k=20 -T=3
python experiment.py --train=True --dataset='caltech' --mode='greedy' --model_folder='models/caltech/greedy/jan30/' -k=20 -T=3
python experiment.py --train=True --dataset='caltech' --mode='random' --model_folder='models/caltech/random/jan30/' -k=20 -T=3

python experiment.py --train=True --dataset='mnist' --mode='vanilla' --model_folder='models/mnist/vanilla/jan30/' -k=10 -T=3
python experiment.py --train=True --dataset='mnist' --mode='greedy' --model_folder='models/mnist/greedy/jan30/' -k=10 -T=3
python experiment.py --train=True --dataset='mnist' --mode='random' --model_folder='models/mnist/random/jan30/' -k=10 -T=3

python experiment.py --train=True --dataset='mnist' --mode='vanilla' --model_folder='models/mnist/vanilla/jan30/' -k=20 -T=3
python experiment.py --train=True --dataset='mnist' --mode='greedy' --model_folder='models/mnist/greedy/jan30/' -k=20 -T=3
python experiment.py --train=True --dataset='mnist' --mode='random' --model_folder='models/mnist/random/jan30/' -k=20 -T=3
