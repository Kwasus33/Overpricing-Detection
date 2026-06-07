DATA_TAGS = ["calendar", "listings"]

SELECTED_FEATURES = {
    "calendar": [
        "listing_id",
        "date",
        "price",  # regression target (as log price)
        # "minimum_nights",  # NaN / empty cells can be imputed by median for selected listing_id
        # "maximum_nights",
        # "available",  # excluded: not a feature, and per-listing booking outcome is the post-hoc heuristic input, not training input
    ],
    "listings": [
        "id",
        # "price",  # excluded as feature: would leak the target (host's stated price)
        # "name",
        # "description",
        # "neighborhood_overview", # all listings from Berlin, Germany
        # "neighbourhood_cleansed",  # districts of Berlin - can filter price in neighbourhoos by that; to one-hot
        "neighbourhood_group_cleansed",  # precise location - e.g. street in district; to one-hot
        "property_type",  # to one-hot
        "room_type",  # to one-hot --> more NaNs in data than in property_type
        # dunno if below needed
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        # "amenities",  # to multi-hot
        # "minimum_nights", # minimum / maximum night can be read from calendar data
        # "maximum_nights",
        # "minimum_minimum_nights",
        # "maximum_minimum_nights",
        # "minimum_maximum_nights",
        # "maximum_maximum_nights",
        # "minimum_nights_avg_ntm",
        # "maximum_nights_avg_ntm",
        # Inference-time unavailable / leakage-adjacent (correlated with `available`):
        # "availability_30",  ### it can be anomaly if listing was booked for most of time during 1 or 2 months and not booked only for few days when price went high
        # "availability_60",
        # "availability_90",
        # "availability_365",
        # Review counts/scores: a new offer has no review history at creation time.
        # "number_of_reviews",
        # "number_of_reviews_ltm",
        # "number_of_reviews_l30d",
        # "first_review",
        # "last_review",
        # "review_scores_rating",  # it's cumulated score, rest of review scores are separated in categories - rest of review_scores used only to get mean value if this is empty
        # "review_scores_accuracy",  # will be dropped
        # "review_scores_cleanliness",  # will be dropped
        # "review_scores_checkin",  # will be dropped
        # "review_scores_communication",  # will be dropped
        # "review_scores_location",  # will be dropped
        # "review_scores_value",  # will be dropped
        # "instant_bookable",  # to encode t==1, f==0
        # "calculated_host_listings_count",
        # "calculated_host_listings_count_entire_homes",
        # "calculated_host_listings_count_private_rooms",
        # "calculated_host_listings_count_shared_rooms",
        # "reviews_per_month",
    ],
}

ONE_HOT_ENCODE = ["neighbourhood_group_cleansed", "property_type", "room_type"]