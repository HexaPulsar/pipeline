# to make this work, we'll have to take the original Mongo Commands, assume they're Ok
# and move on with that 

class ValidCommands:
    # insert + object
    insert_object = "insert_object"
    # update + detections
    insert_detections = "insert_detections"
    # update_probabilities + probabilities in data
    upsert_probabilities = "upsert_probabilities"
    # update_features + features in data
    upsert_features = "upsert_features"
    # update + non_detections
    upsert_non_detections = "upsert_non_detections"


