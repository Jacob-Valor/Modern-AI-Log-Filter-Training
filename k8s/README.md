# Kubernetes manifests

This directory ships the LogFilter application manifests: Deployments,
Services for the in-cluster workloads (API, router, collector, archive),
ingress, NetworkPolicies, secrets, and PVCs.

## Required external services

The `configmap.yaml` and application Deployments reference two services
that are **not** defined in this directory:

- `logfilter-kafka:9092` — message bus between collector, archive,
  router, and the API.
- `logfilter-elasticsearch:9200` — chain-of-custody archive and
  search backplane for the LEEF `raw_log_ref` references.

You must provide one of the following before `kubectl apply -f k8s/`:

1. **External managed services** — point `logfilter-kafka:9092` and
   `logfilter-elasticsearch:9200` at your managed endpoints using a
   `Service` with `type=ExternalName` (or `type=ClusterIP` + custom
   `EndpointSlice`):

   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: logfilter-kafka
     namespace: logfilter
   spec:
     type: ExternalName
     externalName: kafka.example.com
     ports:
       - port: 9092
   ```

   Or use the operator/Helm chart of your provider (Confluent, MSK,
   Elastic Cloud, etc.) and ensure the resulting Service name matches.

2. **In-cluster StatefulSets** — drop in the Bitnami Kafka chart and
   the Elastic Helm chart (or your preferred operators) and verify the
   resulting Service names match the references above. The NetworkPolicies
   in this directory already allow traffic to pods labelled
   `app.kubernetes.io/name=logfilter-kafka` and
   `app.kubernetes.io/name=logfilter-elasticsearch`, so you can either
   match the names or adjust the NetworkPolicy selectors.

## Secrets

`secret.yaml` ships with `REPLACE-ME` placeholders. Set real values
before applying, or replace this file with an `ExternalSecret`,
`SealedSecret`, or a `ServiceAccount`-bound IRSA/Workload Identity token
in your cluster.

## Network policies

`network-policies.yaml` defaults to deny-all egress and selective
ingress. The default `allow-api-ingress` allows traffic from
`logfilter-router` pods, the `monitoring` namespace, and
`ingress-nginx`. Adjust the ingress rule if you use a different
controller (Traefik, Istio, etc.) — see the comment in that file for
the selector to override.

## Ingress

`ingress.yaml` is a placeholder (`api.yourdomain.com`). Set the real
hostname, TLS secret, and controller class before applying. The
NetworkPolicy allowance above assumes an `ingress-nginx` controller
namespace; adjust to match your cluster.
