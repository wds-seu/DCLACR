for lambda in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  OMEGA="model.rdropout.lambda=${lambda}"
  echo $OMEGA
  python3 main.py --hp hparams/nary.rd.yaml --exp "nary.rd.${lambda}" --version v0 --gpus 1 \
                  --omega "$OMEGA"
done
