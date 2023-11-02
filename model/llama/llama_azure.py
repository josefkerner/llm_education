import os
import time
import json
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)
from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ClientSecretCredential,
)
from azure.ai.ml.entities import AmlCompute

from model.model import Model
from typing import Dict, List

class LlamaAzure(Model):

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.get_clients()

    def get_clients(self):
        try:
            credential = DefaultAzureCredential()
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            credential = InteractiveBrowserCredential()

        workspace_ml_client = MLClient(
            credential,
            subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=os.environ["AZURE_RESOURCE_GROUP_NAME"],
            workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
        )
        # the models, fine tuning pipelines and environments are available in the AzureML system registry, "azureml"
        registry_ml_client = MLClient(credential, registry_name="azureml")
        self.workspace_ml_client = workspace_ml_client
        self.registry_ml_client = registry_ml_client

    def get_model_name(self):
        '''
        Get the model name from the registry
        :return:
        '''

        version_list = list(self.registry_ml_client.models.list(self.cfg['model_name']))
        if len(version_list) == 0:
            raise ValueError("Model not found in registry")
        else:
            model_version = version_list[0].version
            foundation_model = self.registry_ml_client.models.get(self.cfg['model_name'], model_version)
            print(
                "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
                    foundation_model.name, foundation_model.version, foundation_model.id
                )
            )
        return foundation_model

    def generate(self, prompts: List[str], temp:float = 0.0):
        '''
        Will generate content
        :param prompts:
        :param temp:
        :return:
        '''
        test_file = "prompts.json"
        sample_json = {"inputs": {"input_string": prompts}}
        with open(test_file, "w") as f:
            json.dump(sample_json, f)

        response = self.workspace_ml_client.online_endpoints.invoke(
            endpoint_name=self.online_endpoint_name,
            deployment_name="demo",
            request_file="prompts.json",
        )
        print("raw response: \n", response, "\n")


    def deploy_model(self):
        foundation_model = self.get_model_name()
        timestamp = int(time.time())
        self.online_endpoint_name = "text-generation-" + str(timestamp)
        # create an online endpoint

        endpoint = ManagedOnlineEndpoint(
            name=self.online_endpoint_name,
            description="Online endpoint for "
                        + foundation_model.name
                        + ", for text-generation task",
            auth_mode="key",
        )
        self.workspace_ml_client.begin_create_or_update(endpoint).wait()

        # create a deployment
        demo_deployment = ManagedOnlineDeployment(
            name="demo",
            endpoint_name=self.online_endpoint_name,
            model=foundation_model.id,
            instance_type="Standard_DS2_v2",
            instance_count=1,
            request_settings=OnlineRequestSettings(
                request_timeout_ms=60000,
            ),
        )
        self.workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
        endpoint.traffic = {"demo": 100}
        self.workspace_ml_client.begin_create_or_update(endpoint).result()
