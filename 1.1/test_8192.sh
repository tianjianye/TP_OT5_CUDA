#!/bin/bash
clear
make clean >/dev/null
make >/dev/null
sh test.sh 8192 100
sh test.sh 8192 500
sh test.sh 8192 1000
sh test.sh 8192 5000
sh test.sh 8192 10000
make clean >/dev/null
