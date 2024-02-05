#!/usr/local/bin/bash
source .env
echo $POSTGRES_USER
kubectl create secret generic postgres-secrets \
  --from-literal=postgres_user=$POSTGRES_USER \
  --from-literal=postgres_password=$POSTGRES_PASSWORD \
  --from-literal=postgres_db=$POSTGRES_DB \
  --dry-run=client -o yaml | kubectl apply -f -