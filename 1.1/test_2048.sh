#!/bin/bash
clear
make clean >/dev/null
make >/dev/null
sh test.sh 2048 100
sh test.sh 2048 500
sh test.sh 2048 1000
sh test.sh 2048 5000
sh test.sh 2048 10000
sh test.sh 2048 50000
make clean >/dev/null
