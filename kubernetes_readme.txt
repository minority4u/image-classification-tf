Deploy as Azure Kubernetes Service

Prerequisites:
-	A complete repository (models currently missing in GitHub)

Steps:
-	Create base-image
	o	docker-compose -f ./docker-compose_base.yml -p multiclasskeras create 
-	Create classification-image
	o	docker-compose -f ./docker-compose.yml create
-	Create Container Registry and push image
	o	Used Azure Container Registry Services: https://docs.microsoft.com/de-de/azure/aks/tutorial-kubernetes-prepare-acr
	o	Pushed base-image first, then classification
			Should work without base-image (needs testing)
-	Create Azure Kubernetes Service
	o	Simply via https://portal.azure.com/ 
	o	Node Size cannot be changed, only scaled!
	o	Create new Service Principal
	o	Use Role-based Access-Control
-	Create/Edit YAML File (kubernetes_deploy.yaml)
	o	image must fit your tagging and registry
-	Create Deployment
	o	kubectl create -f kubernetes_deploy.yaml 
-	Get Public IP
	o	kubectl get service classification-deployment –watch
-	Use Service

Good to Know:
-	Force update of deployment
	o	kubectl replace --force -f kubernetes_deploy.yaml
-	Login to container registry
	o	az acr login --name <acrName>
