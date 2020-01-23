#!/bin/bash
clear
make clean >/dev/null
make >/dev/null
sh test.sh 512 100
sh test.sh 512 500
sh test.sh 512 1000
sh test.sh 512 5000
sh test.sh 512 10000
sh test.sh 512 50000
make clean >/dev/null
