for temperature in 0.01 0.05 0.10 0.5
do
  for lambda in 1.0 0.5 0.25 0.10
  do
    OMEGA="model.contrastive_learning.temperature=${temperature} model.contrastive_learning.lambda=${lambda}"
    echo $OMEGA
    python3 main.py --hp hparams/nary.cl.yaml --exp "nary.cl.${temperature}.${lambda}" --version v0 --gpus 1 \
                    --omega "$OMEGA"
  done
done
