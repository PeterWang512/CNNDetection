#!/bin/bash
DOCUMENT_ID="1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh"
FINAL_DOWNLOADED_FILENAME="trainset.zip"

curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$DOCUMENT_ID" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > $FINAL_DOWNLOADED_FILENAME
unzip trainset.zip
rm trainset.zip
