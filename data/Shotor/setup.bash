#!/usr/bin/bash


FILE='Shotor_Images.tar.gz'
LINK='https://github.com/amirabbasasadi/Shotor/raw/refs/heads/master/Shotor_Images.tar.gz'
[ -f "$FILE" ] || wget -O "$FILE" "$LINK"
[ -f Shotor_Images/0.tif ] || tar -xf "$FILE" &


FILE='Shotor_Words.csv'
LINK='https://github.com/amirabbasasadi/Shotor/raw/refs/heads/master/Shotor_Words.csv'
[ -f "$FILE" ] || wget -O "$FILE" "$LINK"
