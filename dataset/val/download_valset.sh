#!/bin/bash
DOCUMENT_ID="1FU7xF8Wl_F8b0tgL0529qg2nZ_RpdVNL"
FINAL_DOWNLOADED_FILENAME="valset.zip"

curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$DOCUMENT_ID" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > $FINAL_DOWNLOADED_FILENAME
unzip valset.zip
rm valset.zip
