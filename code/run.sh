python run_experiments.py  --dataset cora --g_type var --hidden_size 16 --wd 0.0005 --dp 0.8 --n_hop 10 --lr 0.05 --num_seeds 30 --ts 1 --feat_soft 0 --general_c 9


python run_experiments.py  --dataset citeseer --g_type var --hidden_size 16 --wd 0.0005 --dp 0.5 --n_hop 10 --lr 0.05 --num_seeds 30 --ts 1 --feat_soft 1 --general_c 29


python run_experiments.py  --dataset pubmed --g_type var --hidden_size 16 --wd 5e-05 --dp 0.2 --n_hop 10 --lr 0.05 --num_seeds 30 --ts 1 --feat_soft 1 --general_c 29

python run_experiments.py  --dataset airport --g_type var --hidden_size 64 --wd 5e-05 --dp 0.2 --n_hop 2 --lr 0.01 --num_seeds 30 --ts 1 --feat_soft 0 --general_c 29

python run_experiments.py  --dataset blogcatalog --g_type var --hidden_size 16 --wd 5e-07 --dp 0.5 --n_hop 10 --lr 0.01 --num_seeds 30 --ts 1 --feat_soft 1 --general_c 9


python run_experiments.py  --dataset flickr --g_type var --hidden_size 64 --wd 5e-07 --dp 0.5 --n_hop 2 --lr 0.01 --num_seeds 30 --ts 1 --feat_soft 0 --general_c 29


python run_experiments.py  --new 1 --dataset amazon-com --g_type var --hidden_size 64 --wd 5e-06 --dp 0.5 --n_hop 5 --lr 0.05 --num_seeds 20  --ms 1 --ts 1 --feat_soft 1 --general_c 19 --num_shuffle 20


python run_experiments.py  --new 1 --dataset amazon-ph --g_type var --hidden_size 64 --wd 5e-05 --dp 0.2 --n_hop 5 --lr 0.01 --num_seeds 20  --ms 1 --ts 1 --feat_soft 1 --general_c 19 --num_shuffle 20


python run_experiments.py  --new 1 --dataset co-cs --g_type var --hidden_size 64 --wd 0.0005 --dp 0.2 --n_hop 5 --lr 0.05 --num_seeds 20  --ms 1 --ts 1 --feat_soft 1 --general_c 9 --num_shuffle 20

python run_experiments.py  --new 1 --dataset co-ph --g_type var --hidden_size 64 --wd 5e-05 --dp 0.8 --n_hop 5 --lr 0.01 --num_seeds 20  --ms 1 --ts 1 --feat_soft 1 --general_c 9 --num_shuffle 20