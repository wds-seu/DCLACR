set -v on
for lr in 5.0e-5 5.0e-4 5.0e-3 5.0e-2
do
  python3 main.py --hp hparams/cdr.base.yaml --exp ${lr} --version v0 --gpus 1 --dataset cdr --omega model.optimization.lr\=${lr}
done
