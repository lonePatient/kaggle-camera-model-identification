python runTrain.py --batch_size=64 --pretrained=True --learning_rate=0.0001 --epochs=100
python runMakePseudo.py --batch_size=64
python runTrain.py --batch_size=64 --learning_rate=0.0001 --epochs=100 --resume=True --use_pseudo=True
python runTest.py --batch_size=64
