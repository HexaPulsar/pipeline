# Usage

## Create Initial Parquet Files

### Prepare EC2

- Create an S3 bucket
- Create IAM Profile for EC2 to access an S3 bucket
- Create an EC2 instance with at least 2 * cores * biggest avro tar GBs
- Recommended m7a.xlarge
- Assign the IAM profile to EC2

### Copy project

'''
scp -i ${SSH_PUB_KEY} -r batch-processing/ ubuntu@${EC2_IP}:/home/ubuntu/
'''

### Install dependencies in EC2

'''
sudo apt update
sudo apt install -y curl unzip python3-poetry
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
'''

### Run /scripts/create_avros_parquets.sh

- Adjust variables in 'initalize_variables' method
- Run script, use as many processes as cores (recommended 4)

## Run Spark

### Prepare EMR

- Create S3 bucket or use same as before for output
- Create IAM Role and Policy for EMR and its EC2 servers to have access to S3
- Create Cluster using r7a.xlarge instances
- NOTE: EC2 user to use is 'hadoop'

### Copy project to Primary instance

'''
scp -i ${SSH_PUB_KEY} -r batch-processing/ hadoop@${EC2_IP}:/home/hadoop/
'''

### Prepare Instance

- NOTE: Instance already has python 3.9
'''
rm poetry.lock
bash scripts/emr/prepare_files.sh
'''

### Run script

- Update executor-cores to number of cores of EC2 task instances
- Update num-executors to number of EC2 task instances
'''
spark-submit \
    --name BatchProcessing \
    --deploy-mode cluster \
    --master yarn \
    --archives dist/python-venv.zip#environment \
    --jars libs/jars/healpix-1.0.jar,libs/jars/minimal_astroide-2.0.0.jar \
    --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.351 \
    --executor-cores 4 \
    --num-executors 4 \
    --executor-memory 5G \
    --driver-memory 20G \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
    --conf park.sql.constraintPropagation.enabled=false \
    --verbose \
    scripts/emr/run_batch_processing.py
'''

### Check logs

'''
yarn logs -applicationId <APPLICATION_ID>
'''
