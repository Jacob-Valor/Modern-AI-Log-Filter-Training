"""Raw-log archive consumer entrypoint."""

from __future__ import annotations

import structlog
from elasticsearch import Elasticsearch

from logfilter.config import load_config
from logfilter.kafka.consumer import ArchiveConsumer
from logfilter.pipeline.archive import LogArchive

logger = structlog.get_logger(__name__)


def main() -> None:
    config = load_config()
    kafka_cfg = config.get("kafka", {})
    topics = kafka_cfg.get("topics", {})
    es_cfg = config.get("elasticsearch", {})

    password = es_cfg.get("password", "") or ""
    if not password:
        raise SystemExit(
            "ES password is unset. Set ES_PASSWORD in the environment or "
            "config.elasticsearch.password before starting the archive consumer."
        )
    archive = LogArchive(
        hosts=es_cfg.get("hosts", ["http://localhost:9200"]),
        index_prefix=es_cfg.get("index_prefix", "raw-logs"),
        username=es_cfg.get("username", "elastic"),
        password=password,
        shards=int(es_cfg.get("index_shards", 1)),
        replicas=int(es_cfg.get("index_replicas", 0)),
    )
    es_client: Elasticsearch = archive.client

    consumer = ArchiveConsumer(
        bootstrap_servers=kafka_cfg.get("bootstrap_servers", "localhost:9092"),
        raw_topic=topics.get("raw_logs", "raw-logs"),
        es_client=es_client,
        index_prefix=es_cfg.get("index_prefix", "raw-logs"),
        batch_size=int(kafka_cfg.get("max_poll_records", 100)),
    )

    logger.info("Archive consumer starting")
    consumer.run()


if __name__ == "__main__":  # pragma: no cover
    main()
