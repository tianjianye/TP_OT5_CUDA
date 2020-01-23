#!/bin/bash
clear
make clean >/dev/null
make >/dev/null
sh test.sh 128 100
sh test.sh 128 500
sh test.sh 128 1000
sh test.sh 128 5000
sh test.sh 128 10000
sh test.sh 128 50000
make clean >/dev/null
