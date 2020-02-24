Deploy as Azure Kubernetes Service

Prerequisites:
- A complete repository (models currently missing in GitHub)
- Azure CLI (at least 2.0.53) installed 
	can be downloaded here: https://docs.microsoft.com/de-de/cli/azure/install-azure-cli?view=azure-cli-latest
- Installed Docker locally

Steps:
- Create base-image
	docker-compose -f ./docker-compose_base.yml -p multiclasskeras build
	
- Create classification-image
	docker-compose -f ./docker-compose.yml build --no-cache
	
- Create Container Registry and push image

	az acr create --resource-group <ResourceGroup> --name <acrName> --sku Basic
	
	Pushed base-image first, then classification
		Should work without base-image (needs testing)
		
- Create Azure Kubernetes Service
	Simply via https://portal.azure.com/ using these options (az possible)
		Create new Service Principal
		Use Role-based Access-Control
		Important: Node Size cannot be changed, only scaled horizontally!
	
- Give AKS Service Principal Access to Container Registry
	az role assignment create --assignee <appId> --scope <acrId> --role Reader
	
- Create/Edit YAML File (kubernetes_deploy.yaml)
	image must fit your tag and registry
	
- Create Deployment
	kubectl create -f kubernetes_deploy.yaml 
	
- Get Public IP
	kubectl get service classification-deployment –watch
	
-Use Service

Good to Know:
- Force update of deployment
	kubectl replace --force -f kubernetes_deploy.yaml
	
- Login to container registry
	az acr login --name <acrName>
	
- Set kubectl context to AKS
	az aks get-credentials --resource-group <ResourceGroup> --name <AKSCluster>
	
- Secure Container Registry afterwards
#!/bin/bash

AKS_RESOURCE_GROUP=myAKSResourceGroup
AKS_CLUSTER_NAME=myAKSCluster
ACR_RESOURCE_GROUP=myACRResourceGroup
ACR_NAME=myACRRegistry

# Get the id of the service principal configured for AKS
CLIENT_ID=$(az aks show --resource-group $AKS_RESOURCE_GROUP --name $AKS_CLUSTER_NAME --query "servicePrincipalProfile.clientId" --output tsv)

# Get the ACR registry resource id
ACR_ID=$(az acr show --name $ACR_NAME --resource-group $ACR_RESOURCE_GROUP --query "id" --output tsv)

# Create role assignment
az role assignment create --assignee $CLIENT_ID --role acrpull --scope $ACR_ID

- For autoscaling the deployment manifest needs metrics
