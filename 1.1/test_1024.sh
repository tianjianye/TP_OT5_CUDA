#!/bin/bash
clear
make clean >/dev/null
make >/dev/null
sh test.sh 1024 100
sh test.sh 1024 500
sh test.sh 1024 1000
sh test.sh 1024 5000
sh test.sh 1024 10000
sh test.sh 1024 50000
make clean >/dev/null
