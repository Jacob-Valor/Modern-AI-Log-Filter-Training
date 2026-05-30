"""Integration test: produce and consume a message through real Kafka."""

from __future__ import annotations

import time
import uuid

import pytest


@pytest.mark.integration
def test_kafka_produce_consume_roundtrip(
    kafka_admin,
    kafka_producer,
    kafka_consumer,
):
    """Create a topic, produce a message, and consume it back."""
    topic = f"test-roundtrip-{uuid.uuid4().hex[:8]}"

    # Ensure topic exists
    from kafka.admin import NewTopic

    kafka_admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])

    # Subscribe
    kafka_consumer.subscribe([topic])

    # Produce
    payload = {"event": "test", "timestamp": time.time()}
    future = kafka_producer.send(topic, value=payload)
    future.get(timeout=10)
    kafka_producer.flush()

    # Consume
    messages = []
    for _ in range(30):
        raw = kafka_consumer.poll(timeout_ms=1000)
        for msgs in raw.values():
            messages.extend(msgs)
        if messages:
            break

    assert len(messages) == 1
    assert messages[0].value == payload


@pytest.mark.integration
def test_kafka_log_producer_integration(kafka_admin, kafka_bootstrap: str):
    """Smoke-test our LogProducer against the real Kafka broker."""
    from logfilter.kafka.producer import LogProducer

    topic = f"test-producer-{uuid.uuid4().hex[:8]}"
    from kafka.admin import NewTopic

    kafka_admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])

    producer = LogProducer(
        bootstrap_servers=kafka_bootstrap,
        topic=topic,
        compression=None,
    )
    envelope = {
        "raw": "<14>test event",
        "host": "localhost",
        "timestamp": time.time(),
        "format": "syslog",
    }
    producer.send(envelope)
    producer.close()

    # Verify via direct consumer
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_bootstrap,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: v.decode("utf-8"),
        consumer_timeout_ms=10000,
    )
    msgs = list(consumer)
    consumer.close()

    assert len(msgs) == 1
    assert "test event" in msgs[0].value
