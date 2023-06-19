import math
from typing import Union, List
from pymongo.errors import BulkWriteError

from db_plugins.db.mongo.connection import DatabaseConnection
from db_plugins.db.mongo.models import Object


def oid_query(db: DatabaseConnection, oid: list) -> Union[str, None]:
    """
    Query the database and check if any of the OIDs is already in database

    :param db: Database connection
    :param oid: oid of any survey

    :return: existing aid if exists else is None
    """
    mongo_query = db.query(Object)
    found = mongo_query.collection.find_one({"oid": {"$in": oid}}, {"_id": 1})
    if found:
        return found["_id"]
    return None


def conesearch_query(
    db: DatabaseConnection, ra: float, dec: float, radius: float
) -> Union[str, None]:
    """
    Query the database and check if there is an alerce_id
    for the specified coordinates and search radius

    :param db: Database connection
    :param ra: first coordinate argument (RA)
    :param dec: first coordinate argument (Dec)
    :param radius: search radius (arcsec)

    :return: existing aid if exists else is None
    """
    mongo_query = db.query(Object)
    found = mongo_query.collection.find_one(
        {
            "loc": {
                "$nearSphere": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [ra - 180, dec],
                    },
                    "$maxDistance": math.radians(radius / 3600),
                },
            },
        },
        {"_id": 1},
    )
    if found:
        return found["_id"]
    return None


def update_query(db: DatabaseConnection, records: List[dict]):
    """
    Insert or update the records in a dictionary. Pushes the oid array to
    oid column.

    :param db: Database connection
    :param records: Records containing _id and oid fields to insert or update
    """
    mongo_query = db.query(Object)
    for record in records:
        query = {"_id": record["_id"]}
        new_value = {
            "$addToSet": {"oid": {"$each": record["oid"]}},
        }
        updated = mongo_query.collection.find_one_and_update(
            query, new_value, upsert=True, return_document=True
        )
