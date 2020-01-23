#!/bin/bash
clear
make clean >/dev/null
make >/dev/null
sh test.sh 256 100
sh test.sh 256 500
sh test.sh 256 1000
sh test.sh 256 5000
sh test.sh 256 10000
sh test.sh 256 50000
make clean >/dev/null
