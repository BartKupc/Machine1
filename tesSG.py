import boto3
import sagemaker
from sagemaker.remote_function import remote

sm_session = sagemaker.Session(boto_session=boto3.session.Session(region_name="eu-west-1"))
settings = dict(
    sagemaker_session=sm_session,
    role = 'arn:aws:iam::688567281415:role/service-role/AmazonSageMaker-ExecutionRole-20240913T093672',
    instance_type="ml.m5.xlarge",
    dependencies='./requirements.txt'
)

@remote(**settings)
def divide(x, y):
    return x / y


if __name__ == "__main__":
    print(divide(2, 3.0))