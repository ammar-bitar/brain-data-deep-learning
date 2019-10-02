#
#
# This script intends to do some tests with the
# MEG data stored in Amazon S3 server, along with 
# the library boto (pip install boto3)
#
#


import boto3
import sys

#Initializing connexion
s3 = boto3.client('s3', aws_access_key_id="AKIAXO65CT57DAAAYBUM", aws_secret_access_key="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")
print("Accesing the data...")
#Getting object
obj = s3.get_object(Bucket='hcp-openaccess', Key='HCP_900/100307/unprocessed/MEG/3-Restin/4D/c,rfDC')
#obj = s3.get_object(Bucket='hcp-openaccess', Key='HCP_900/100307/unprocessed/MEG/3-Restin/4D/config')
#Extracting data
data = obj['Body'].read()

print(data)

print("Success")