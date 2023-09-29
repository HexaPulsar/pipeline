import json
import os
import pytest
import unittest

from sqlalchemy import text

from mongo_scribe.step import MongoScribe

from apf.producers.kafka import KafkaProducer
from db_plugins.db.sql._connection import PsqlDatabase

DB_CONFIG = {
    "PSQL": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "PORT": 5432,
        "DB_NAME": "postgres",
    }
}

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "TOPICS": ["test_topic"],
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "command_consumer_2",
        "enable.partition.eof": True,
        "auto.offset.reset": "beginning",
    },
    "NUM_MESSAGES": 2,
    "TIMEOUT": 10,
}

PRODUCER_CONFIG = {
    "TOPIC": "test_topic",
    "PARAMS": {"bootstrap.servers": "localhost:9092"},
    "SCHEMA": {
        "namespace": "db_operation",
        "type": "record",
        "name": "Command",
        "fields": [
            {"name": "payload", "type": "string"},
        ],
    },
}


@pytest.mark.usefixtures("psql_service")
@pytest.mark.usefixtures("kafka_service")
class MongoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = PsqlDatabase(DB_CONFIG["PSQL"])
        step_config = {
            "DB_CONFIG": DB_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
        }
        cls.db.create_db()
        with cls.db.session() as session:
            session.execute(
                text(
                    """
                    INSERT INTO step(step_id, name, version, comments, date) 
                    VALUES ('v1', 'version 1', '1', '', current_timestamp)
                    """
                )
            )
            session.commit()
        cls.step = MongoScribe(config=step_config, db="sql")
        cls.producer = KafkaProducer(config=PRODUCER_CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.db.drop_db()

    def test_insert_objects_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {
                    "oid": "ZTF02ululeea",
                    "ndet": 1,
                    "firstmjd": 50001,
                    "g_r_max": 1.0,
                    "g_r_mean_corr": 0.9,
                    "meanra": 45,
                    "meandec": 45,
                },
            }
        )
        command2 = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {
                    "oid": "ZTF03ululeea",
                    "ndet": 1,
                    "firstmjd": 50001,
                    "g_r_max": 1.0,
                    "g_r_mean_corr": 0.9,
                    "meanra": 45,
                    "meandec": 45,
                },
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command2})
        self.producer.producer.flush(1)
        self.step.start()
        with self.db.session() as session:
            result = session.execute(text("SELECT * FROM object"))
            oids = [r[0] for r in result]
            assert "ZTF02ululeea" in oids
            assert "ZTF03ululeea" in oids

    def test_insert_detections_into_database(self):
        command = json.dumps(
            {
                "collection": "detection",
                "type": "update",
                "data": {
                    "candid": 932472823,
                    "oid": "ZTF04ululeea",
                    "mjd": 55000,
                    "fid": 1,
                    "pid": 4.3,
                    "isdiffpos": 1,
                    "ra": 99.0,
                    "dec": 55.0,
                    "magpsf": 220.0,
                    "sigmapsf": 33.0,
                    "corrected": True,
                    "dubious": True,
                    "has_stamp": True,
                    "step_id_corr": "steppu",
                },
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command})

        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        self.step.start()

        with self.db.session() as session:
            result = session.execute(
                text(
                    """ SELECT candid, oid FROM detection WHERE oid = 'ZTF04ululeea' """
                )
            )
            result = list(result)[0]
            assert result[0] == 932472823
            assert result[1] == "ZTF04ululeea"

    def test_upsert_non_detections(self):
        command = {
            "collection": "non_detection",
            "type": "update",
            "criteria": {
                "aid": "AL21XXX",
                "oid": "ZTF04ululeea",
                "mjd": 55000,
                "fid": 1,
            },
            "data": {
                "diffmaglim": 0.1,
                "pid": 4.3,
                "isdiffpos": 1,
                "ra": 99.0,
                "dec": 55.0,
                "magpsf": 220.0,
                "sigmapsf": 33.0,
                "step_id_corr": "steppu",
            },
        }

        self.producer.produce({"payload": json.dumps(command)})
        command["criteria"]["fid"] = 2
        self.producer.produce({"payload": json.dumps(command)})

        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        self.step.start()

        with self.db.session() as session:
            result = session.execute(
                text(
                    """ SELECT fid, mjd, diffmaglim FROM non_detection WHERE oid = 'ZTF04ululeea' """
                )
            )
            result = list(result)
            assert len(result) == 2
            assert result[0][0] == 1
            assert result[0][1] == 55000
            assert result[0][2] == 0.1

    def test_upsert_features(self):
        with self.db.session() as session:
            session.execute(
                text(
                    """
                    INSERT INTO feature_version(version, step_id_feature, step_id_preprocess)
                    VALUES ('1.0.0', 'v1', 'v1')
                    """
                )
            )
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        command_data = [
            {
                "name": "feature1",
                "value": 55.0,
                "fid": 1,
            },
            {
                "name": "feature2",
                "value": 22.0,
                "fid": 2,
            },
            {
                "name": "feature3",
                "value": 130.0,
                "fid": 1,
            },
            {
                "name": "feature1",
                "value": 694211.0,
                "fid": 1,
            },
        ]

        commands = [
            {
                "payload": json.dumps(
                    {
                        "collection": "object",
                        "type": "update_features",
                        "criteria": {
                            "_id": "AL21XXX",
                            "oid": ["ZTF04ululeea"]
                        },
                        "data": {
                            "features_version": "1.0.0",
                            "features_group": "ztf",
                            "features": [data],
                        },
                    }
                )
            }
            for data in command_data
        ]

        for command in commands:
            self.producer.produce(command)
        self.producer.producer.flush(4)
        self.step.start()
        with self.db.session() as session:
            result = session.execute(
                text(
                    """
                    SELECT name, value FROM feature WHERE oid = 'ZTF04ululeea' AND name = 'feature1'
                """
                )
            )
            result = list(result)[0]
            assert result[0] == "feature1"
            assert result[1] == 694211.0

    def test_upsert_probabilities(self):
        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        command_data = [
            {
                "class_name": "class1",
                "classifier_name": "lc_classifier",
                "classifier_version": "1.0.0",
                "probability": 0.6,
                "ranking": 1,
            },
            {
                "class_name": "class2",
                "classifier_name": "lc_classifier",
                "classifier_version": "1.0.0",
                "probability": 0.4,
                "ranking": 2,
            },
            {
                "class_name": "class2",
                "classifier_name": "lc_classifier",
                "classifier_version": "1.0.0",
                "probability": 0.65,
                "ranking": 1,
            },
            {
                "class_name": "class1",
                "classifier_name": "lc_classifier",
                "classifier_version": "1.0.0",
                "probability": 0.35,
                "ranking": 2,
            },
        ]

        commands = [
            {
                "payload": json.dumps(
                    {
                        "collection": "object",
                        "type": "update_probabilities",
                        "criteria": {
                            "_id": "AL21XXX",
                            "oid": ["ZTF04ululeea"]
                        },
                        "data": data,
                    }
                )
            }
            for data in command_data
        ]

        for command in commands:
            self.producer.produce(command)

        self.producer.producer.flush(4)
        self.step.start()

        with self.db.session() as session:
            result = session.execute(
                text(
                    """
                    SELECT * FROM probability WHERE oid = 'ZTF04ululeea'
                    """
                )
            )
            result = list(result)
            assert len(result) == 2
            for row in result:
                if row[1] == "class2":
                    assert row[4] == 0.65 and row[5] == 1
                else:
                    assert row[4] == 0.35 and row[5] == 2

    def test_update_object_stats(self):
        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        command = {
            "collection": "magstats",
            "criteria": {"oid": "ZTF04ululeea"},
            "data": {
                "corrected": False,
                "firstmjd": 0.04650036190916984,
                "lastmjd": 0.9794581336685745,
                "magstats": [
                    {
                        "corrected": False,
                        "dm_first": None,
                        "dmdt_first": None,
                        "dt_first": None,
                        "fid": "r",
                        "firstmjd": 0.04650036190916984,
                        "lastmjd": 0.9794581336685745,
                        "magfirst": 0.45738787694615635,
                        "magfirst_corr": None,
                        "maglast": 0.8891703032055938,
                        "maglast_corr": None,
                        "magmax": 0.9954724098279284,
                        "magmax_corr": None,
                        "magmean": 0.6485098616306181,
                        "magmean_corr": None,
                        "magmedian": 0.6183493589106022,
                        "magmedian_corr": None,
                        "magmin": 0.29146111487295745,
                        "magmin_corr": None,
                        "magsigma": 0.24471928116997924,
                        "magsigma_corr": None,
                        "ndet": 9,
                        "ndubious": 0,
                        "saturation_rate": None,
                        "sid": "ZTF",
                        "sigmadm_first": None,
                        "stellar": False,
                    },
                ],
                "meandec": 0.4861642021396574,
                "meanra": 0.5267988555440914,
                "ndet": 20,
                "sigmadec": 0.00568264450571807,
                "sigmara": 0.0006830686562186637,
                "stellar": False,
            },
            "type": "upsert",
        }

        self.producer.produce({"payload": json.dumps(command)})

        command["data"]["magstats"] = [
            {
                "corrected": False,
                "dm_first": None,
                "dmdt_first": None,
                "dt_first": None,
                "fid": "g",
                "firstmjd": 0.13577030206791907,
                "lastmjd": 0.95383888383811,
                "magfirst": 0.6249465481253661,
                "magfirst_corr": None,
                "maglast": 0.894922004401134,
                "maglast_corr": None,
                "magmax": 0.894922004401134,
                "magmax_corr": None,
                "magmean": 0.4860666136917287,
                "magmean_corr": None,
                "magmedian": 0.6062813119154207,
                "magmedian_corr": None,
                "magmin": 0.03844908454164819,
                "magmin_corr": None,
                "magsigma": 0.2650409061639637,
                "magsigma_corr": None,
                "ndet": 11,
                "ndubious": 0,
                "saturation_rate": None,
                "sid": "ZTF",
                "sigmadm_first": None,
                "stellar": False,
            }
        ]
        self.producer.produce({"payload": json.dumps(command)})
        self.producer.producer.flush(1)
        self.step.start()

        with self.db.session() as session:
            result = session.execute(
                text(
                    """
                    SELECT object.oid as oid, fid, meanra, meandec, magstat.ndet as ndet
                    FROM object JOIN magstat on object.oid = magstat.oid 
                    """
                )
            )
            print(list(result))
