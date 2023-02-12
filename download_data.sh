#!/bin/bash

while [ $# -gt 0 ]; do
case "$1" in
--data-folder=)
data_folder="${1#=}"
;;
*)
echo "Unknown option: $1"
exit 1
;;
esac
shift
done

if [ -z "$data_folder" ]; then
echo "Error: --data-folder argument is required."
exit 1
fi

mkdir -p "$data_folder"
aicrowd login
aicrowd dataset download -c esci-challenge-for-improving-product-search -o "$data_folder" 0 1 2 3
unzip "$data_folder/*.zip"




