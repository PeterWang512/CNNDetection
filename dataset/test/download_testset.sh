#!/bin/bash
DOCUMENT_ID="1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1"
FINAL_DOWNLOADED_FILENAME="testset.zip"

curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$DOCUMENT_ID" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > $FINAL_DOWNLOADED_FILENAME
unzip testset.zip
rm testset.zip
