# Kubernetes (local cluster only)

Tested target: minikube. This is a learning setup, not a production cluster:
storage is a hostPath volume and there is no ingress, autoscaling or backup.

```bash
minikube start --cpus=4 --memory=4096
eval $(minikube docker-env)                      # build inside minikube
docker build -t credit-data-pipeline:local .

kubectl apply -f k8s/namespace.yaml              # 1. namespace first
kubectl -n credit-pipeline create secret generic credit-pipeline-db \
  --from-literal=DB_NAME=credit_risk_db \
  --from-literal=DB_USER=dataengineer \
  --from-literal=DB_PASSWORD="$(openssl rand -hex 16)"   # 2. secret, never committed
kubectl apply -f k8s/postgres.yaml               # 3. database
minikube ssh -- sudo mkdir -p /data/credit-pipeline/raw
minikube cp data/raw/cs-training.csv /data/credit-pipeline/raw/cs-training.csv
kubectl apply -f k8s/job.yaml                    # 4. run the pipeline once
kubectl -n credit-pipeline logs -f job/credit-pipeline
```

Why these choices:

| Decision | Reason |
|---|---|
| `Job`, not `Deployment` | The pipeline finishes. A Deployment would restart it in a loop. |
| Secret created with `kubectl` | Base64 in a committed YAML is encoding, not encryption. |
| Namespace in its own file, applied first | `kubectl apply` processes documents in order; objects in a namespace that does not exist yet fail. |
