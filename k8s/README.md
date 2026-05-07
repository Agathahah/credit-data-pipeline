# Kubernetes Deployment — Credit Data Pipeline

Production-grade Kubernetes configuration for deploying the credit risk ML pipeline on a multi-node cluster.

## Architecture

```
ml-production namespace
├── postgres (Deployment + Service)
│   └── PersistentVolumeClaim (10Gi)
└── credit-pipeline (Deployment)
    ├── PersistentVolumeClaim: data (5Gi)
    ├── PersistentVolumeClaim: models (1Gi)
    └── initContainer: wait-for-postgres
```

## Files

| File | Description |
|---|---|
| `deployment.yaml` | Pipeline container spec with resource limits and init container |
| `service.yaml` | PostgreSQL Service, Deployment, Secrets, PVCs, and Namespace |

## Prerequisites

- Kubernetes cluster (local: minikube / kind, cloud: GKE / EKS / AKS)
- kubectl configured
- Docker image pushed to registry

## Deploy

```bash
# Apply all configurations
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment.yaml

# Verify
kubectl get pods -n ml-production
kubectl get pvc -n ml-production
kubectl logs -f deployment/credit-pipeline -n ml-production
```

## Configuration

Credentials are managed via Kubernetes Secret. To update:

```bash
# Encode new value
echo -n "new_password" | base64

# Edit secret
kubectl edit secret credit-pipeline-secret -n ml-production
```

## Resource Allocation

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---|---|---|---|---|
| credit-pipeline | 250m | 1000m | 512Mi | 2Gi |
| postgres | 100m | 500m | 256Mi | 1Gi |
