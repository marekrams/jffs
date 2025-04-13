#!/bin/bash

# nohup taskset -c 0-8 python -u run_evolution.py -mm 0.4 -D0 1024 -a 0.0625 -N 1024 > out4.txt &
# nohup taskset -c 9-17 python -u run_evolution.py -mm 0.6 -D0 1024 -a 0.0625 -N 1024 > out6.txt &
# nohup taskset -c 18-26 python -u run_evolution.py -mm 0.7 -D0 1024 -a 0.0625 -N 1024 > out7.txt &
# nohup taskset -c 27-35 python -u run_evolution.py -mm 0.8 -D0 1024 -a 0.0625 -N 1024 > out8.txt &

# # nohup taskset -c 36-44 python -u run_evolution.py -mm 0.4 -D0 1024 -a 0.0625 -N 1024 > out4.txt &
# nohup taskset -c 36-44 python -u run_evolution.py -mm 1.0 -D0 1024 -a 0.0625 -N 1024 > out10.txt &
# nohup taskset -c 45-53 python -u run_evolution.py -mm 1.1 -D0 1024 -a 0.0625 -N 1024 > out11.txt &
# nohup taskset -c 54-62 python -u run_evolution.py -mm 1.2 -D0 1024 -a 0.0625 -N 1024 > out12.txt &

nohup taskset -c 63-71 python -u run_temp.py -mm 0.5 -D0 512 -a 0.0625 -N 128 > out_temp.txt &

# nohup taskset -c 28-29 python -u run_evolution.py -mm 0.0 -D0 256 -a 0.16666666666 -N 384 > outn0.txt &
# nohup taskset -c 30-31 python -u run_evolution.py -mm 0.1 -D0 256 -a 0.16666666666 -N 384 > outn1.txt &
# nohup taskset -c 32-33 python -u run_evolution.py -mm 0.2 -D0 256 -a 0.16666666666 -N 384 > outn2.txt &
# nohup taskset -c 34-35 python -u run_evolution.py -mm 0.318309886 -D0 256 -a 0.16666666666 -N 384 > outn3.txt &
# nohup taskset -c 36-37 python -u run_evolution.py -mm 0.4 -D0 256 -a 0.16666666666 -N 384 > outn4.txt &
# nohup taskset -c 38-39 python -u run_evolution.py -mm 0.5 -D0 256 -a 0.16666666666 -N 384 > outn5.txt &
# nohup taskset -c 40-41 python -u run_evolution.py -mm 0.6 -D0 256 -a 0.16666666666 -N 384 > outn6.txt &
# nohup taskset -c 42-43 python -u run_evolution.py -mm 0.7 -D0 256 -a 0.16666666666 -N 384 > outn7.txt &
# nohup taskset -c 54-55 python -u run_evolution.py -mm 0.8 -D0 256 -a 0.16666666666 -N 384 > outn8.txt &

# nohup taskset -c 18-19 python -u run_evolution.py -mm 0.9 -D0 256 -a 0.16666666666 -N 384 > outn9.txt &
# nohup taskset -c 20-21 python -u run_evolution.py -mm 1.0 -D0 256 -a 0.16666666666 -N 384 > outn10.txt &
# nohup taskset -c 22-23 python -u run_evolution.py -mm 1.1 -D0 256 -a 0.16666666666 -N 384 > outn11.txt &
# nohup taskset -c 24-25 python -u run_evolution.py -mm 1.2 -D0 256 -a 0.16666666666 -N 384 > outn12.txt &
# nohup taskset -c 56-57 python -u run_evolution.py -mm 1.5 -D0 256 -a 0.16666666666 -N 384 > outn11.txt &
# nohup taskset -c 58-59 python -u run_evolution.py -mm 2.0 -D0 256 -a 0.16666666666 -N 384 > outn12.txt &

# nohup taskset -c 36-37 python -u run_evolution.py -mm 0.0 -D0 256 -a 0.03125 -N 512 -Q 10 > out4.txt &
# nohup taskset -c 38-39 python -u run_evolution.py -mm 0.1 -D0 256 -a 0.03125 -N 512 -Q 10 > out5.txt &
# nohup taskset -c 40-41 python -u run_evolution.py -mm 0.2 -D0 256 -a 0.03125 -N 512 -Q 10 > out6.txt &
# nohup taskset -c 42-43 python -u run_evolution.py -mm 0.3 -D0 256 -a 0.03125 -N 512 -Q 10 > out7.txt &
# nohup taskset -c 44-45 python -u run_evolution.py -mm 0.4 -D0 256 -a 0.03125 -N 512 -Q 10 > out8.txt &
# nohup taskset -c 46-47 python -u run_evolution.py -mm 0.5 -D0 256 -a 0.03125 -N 512 -Q 10 > out9.txt &
# nohup taskset -c 48-49 python -u run_evolution.py -mm 0.6 -D0 256 -a 0.03125 -N 512 -Q 10 > out10.txt &
# nohup taskset -c 50-51 python -u run_evolution.py -mm 0.7 -D0 256 -a 0.03125 -N 512 -Q 10 > out11.txt &
# nohup taskset -c 52-53 python -u run_evolution.py -mm 0.8 -D0 256 -a 0.03125 -N 512 -Q 10 > out12.txt &
# nohup taskset -c 54-55 python -u run_evolution.py -mm 0.9 -D0 256 -a 0.03125 -N 512 -Q 10 > out13.txt &
# nohup taskset -c 56-57 python -u run_evolution.py -mm 1.0 -D0 256 -a 0.03125 -N 512 -Q 10 > out14.txt &
# nohup taskset -c 58-59 python -u run_evolution.py -mm 1.1 -D0 256 -a 0.03125 -N 512 -Q 10 > out15.txt &
# nohup taskset -c 60-61 python -u run_evolution.py -mm 1.2 -D0 256 -a 0.03125 -N 512 -Q 10 > out16.txt &
# nohup taskset -c 62-63 python -u run_evolution.py -mm 1.5 -D0 256 -a 0.03125 -N 512 -Q 10 > out17.txt &
# nohup taskset -c 64-65 python -u run_evolution.py -mm 2.0 -D0 256 -a 0.03125 -N 512 -Q 10 > out18.txt &