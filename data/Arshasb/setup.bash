#!/usr/bin/bash

FILE='Arshasb_7k.tar.gz'
LINK='https://drive.usercontent.google.com/download?id=1G9JEZY9MSzaND8ynnFodIXQvMMM1_6J3&export=download&authuser=0&confirm=t&uuid=b54052f6-2f0a-4d08-9eae-7c55d6e015ae&at=AIrpjvMMzg790Ax5fLRz3hmZ33B-%3A1737638100634'
[ -f "$FILE" ] || wget -O "$FILE" "$LINK"
[ -f Shotor_Images/0.tif ] || tar -xf "$FILE" &
