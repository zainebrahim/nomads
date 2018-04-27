from flask import Flask, flash, render_template, url_for, \
request, redirect
import boto3

app = Flask(__name__)

def submit_job(token, col, exp, z_range, y_range, x_range):
    client = boto3.client("batch")
    response = client.describe_compute_environments(
        computeEnvironments=[
            'nomads-ce',
        ],
    )
    if len(response["computeEnvironments"]) == 0:
        response = client.create_compute_environment(
            type='MANAGED',
            computeEnvironmentName='nomads-ce',
            computeResources={
                'type': 'EC2',
                'desiredvCpus': 0,
                'instanceRole': 'ecsInstanceRole',
                'instanceTypes': [
                    "optimal"
                ],
                'maxvCpus': 20,
                'minvCpus': 0,
                'securityGroupIds': [
                    'sg-41927a3e',
                ],
                'subnets': [
                    'subnet-11dc531d',
                    'subnet-17c65f72',
                    'subnet-75006549',
                    'subnet-4e2ace06',
                    'subnet-7ca59151',
                    'subnet-74f3d02f'
                ],
                'tags': {
                    'Name': 'Batch Instance - C4OnDemand',
                },
            },
            serviceRole='arn:aws:iam::389826612951:role/service-role/AWSBatchServiceRole',
            state='ENABLED',
        )
    
    response = client.describe_job_queues(jobQueues=["nomads-queue"])
    if len(response["jobQueues"]) == 0:
        response = client.create_job_queue(
            jobQueueName='string',
            state='ENABLED',
            priority=10,
            computeEnvironmentOrder=[
                {
                    'order': 1,
                    'computeEnvironment': 'nomads-ce'
                },
            ]
        )
    
    response = client.describe_job_definitions(
        jobDefinitionName='nomads-unsupervised',
        status='ACTIVE',
    )
    if len(response["jobDefinitions"]) == 0:
        response = client.register_job_definition(
            type='container',
            containerProperties={
                'command': [
                    "echo",
                    "Staring Container"
                ],
                'image': '389826612951.dkr.ecr.us-east-1.amazonaws.com/nomads-unsupervised',
                'memory': 4000,
                'vcpus': 1,
            },
            jobDefinitionName='nomads-unsupervised',
        )
    job_name = "-".join([col, exp, z_range, y_range, x_range])
    job_name = job_name.replace(",", "-")
    response = client.submit_job(
        jobName=job_name,
        jobQueue='nomads-queue',
        jobDefinition='nomads-unsupervised',
        containerOverrides={
            'vcpus': 1,
            'memory': 2000,
            'command': [
                "python3",
                "driver.py",
                "--host",
                "api.boss.neurodata.io",
                "--token",
                token,
                "--col",
                col,
                "--exp",
                exp,
                "--z-range",
                z_range,
                "--x-range",
                x_range,
                "--y-range",
                y_range
            ],
        },
    )
    return
    
            
@app.route("/", methods = ["GET"])
def index():
    return render_template("index.html")

@app.route("/submit", methods = ["GET", "POST"])
def submit():
    token = request.form["token"]
    col = request.form["col"]
    exp = request.form["exp"]
    z_range = request.form["z_range"].replace(" ", "")
    y_range = request.form["y_range"].replace(" ", "")
    x_range = request.form["x_range"].replace(" ", "")
    host = "api.boss.neurodata.io"
    submit_job(token, col, exp, z_range, y_range, x_range)
    return redirect(url_for("index"))

if __name__ == "__main__":
    #submit_job("edef359a8de270163c911dcef5d467a72348d68d", "collman", "M247514_Rorb_1_light", "40,45", "6500,7000", "6500,7000")
    app.run(debug = True, port = 8000)

