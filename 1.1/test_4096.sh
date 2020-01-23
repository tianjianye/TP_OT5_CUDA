#!/bin/bash
clear
make clean >/dev/null
make >/dev/null
sh test.sh 4096 100
sh test.sh 4096 500
sh test.sh 4096 1000
sh test.sh 4096 5000
sh test.sh 4096 10000
make clean >/dev/null
