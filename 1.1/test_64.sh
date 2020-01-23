#!/bin/bash
clear
make clean >/dev/null
make >/dev/null
sh test.sh 64 10
sh test.sh 64 100
sh test.sh 64 500
sh test.sh 64 1000
sh test.sh 64 5000
sh test.sh 64 10000
sh test.sh 64 50000
make clean >/dev/null
