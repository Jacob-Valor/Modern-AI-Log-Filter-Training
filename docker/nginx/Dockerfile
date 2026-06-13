FROM nginxinc/nginx-unprivileged:1.27-alpine@sha256:65e3e85dbaed8ba248841d9d58a899b6197106c23cb0ff1a132b7bfe0547e4c0

USER root
RUN apk add --no-cache curl
RUN rm /etc/nginx/conf.d/default.conf
COPY config/nginx/nginx.conf /etc/nginx/conf.d/logfilter.conf
RUN chown -R 101:101 /etc/nginx/conf.d
USER 101

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -fsk https://localhost:8443/health || exit 1

EXPOSE 8080 8443
