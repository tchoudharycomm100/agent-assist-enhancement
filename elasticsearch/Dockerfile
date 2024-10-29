FROM docker.elastic.co/elasticsearch/elasticsearch:8.15.3
ENV  discovery.type=single-node
ENV xpack.security.enabled=false
USER root
RUN chown -R elasticsearch:elasticsearch /usr/share/elasticsearch/config /usr/share/elasticsearch/data /usr/share/elasticsearch/logs
USER elasticsearch
EXPOSE 9200 9300
CMD ["bin/elasticsearch"]