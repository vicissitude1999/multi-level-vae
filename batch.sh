#!/bin/bash

#python3 main.py --encoder_save encoder_n=100_T=50 --decoder_save decoder_n=100_T=50
#python3 main.py --encoder_save encoder_n=200_T=50 --decoder_save decoder_n=200_T=50
#python3 main.py --encoder_save encoder_n=500_T=50 --decoder_save decoder_n=500_T=50
#python3 main.py --encoder_save encoder_n=1000_T=50 --decoder_save decoder_n=1000_T=50

python3 main.py --encoder_save encoder_n=100_T=500 --decoder_save decoder_n=100_T=500
python3 main.py --encoder_save encoder_n=200_T=500 --decoder_save decoder_n=200_T=500
python3 main.py --encoder_save encoder_n=500_T=500 --decoder_save decoder_n=500_T=500
python3 main.py --encoder_save encoder_n=1000_T=500 --decoder_save decoder_n=1000_T=500