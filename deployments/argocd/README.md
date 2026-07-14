# ArgoCD Application definition

This folder contains the ArgoCD Application definitions used to deploy the Enterprise Knowledge Hub. These are now self-contained in this repository (previously split across the `ssc-dsai-iac` infrastructure repositories):

* `application.yaml` - base Application definition.
* `local-application.yaml` - local/minikube Application, using `deployments/ekh/values-local.yaml`.
* `dev-application.yaml` - AKS dev environment Application, using `deployments/ekh/values-dev.yaml`.

There is currently no production environment configured; it will be added here once available.