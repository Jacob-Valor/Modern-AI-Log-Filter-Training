FROM nginx:1.27-alpine

RUN rm /etc/nginx/conf.d/default.conf
COPY config/nginx/nginx.conf /etc/nginx/conf.d/logfilter.conf

EXPOSE 80 443
