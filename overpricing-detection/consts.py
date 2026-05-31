# TODO: check in notebook data analysis which columns are e.g. more than half empty or all empty (like 'calendar updated') or have const value (like has_availability) and drop them from features on which models will be trained

DATA_TAGS = ["calendar", "listings"]

SELECTED_FEATURES = {
    "calendar": [
        "listing_id",
        "date",
        "available",
        "price",
        # dunno if needed
        # "minimum_nights",  # NaN / empty cells can be imputed by median for selected listing_id
        # "maximum_nights",
    ],
    "listings": [
        "id",
        "price",  # strip price from $100.00 format
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
        "amenities",  # to multi-hot
        # "minimum_nights", # minimum / maximum night can be read from calendar data
        # "maximum_nights",
        # "minimum_minimum_nights",
        # "maximum_minimum_nights",
        # "minimum_maximum_nights",
        # "maximum_maximum_nights",
        # "minimum_nights_avg_ntm",
        # "maximum_nights_avg_ntm",
        "availability_30",  ### it can be anomaly if listing was booked for most of time during 1 or 2 months and not booked only for few days when price went high
        "availability_60",
        # "availability_90",
        # "availability_365",
        # "number_of_reviews",
        "number_of_reviews_ltm",
        "number_of_reviews_l30d",
        # "first_review",
        # "last_review",
        "review_scores_rating",  # it's cumulated score, rest of review scores are separated in categories - rest of review_scores used only to get mean value if this is empty
        "review_scores_accuracy",  # will be dropped
        "review_scores_cleanliness",  # will be dropped
        "review_scores_checkin",  # will be dropped
        "review_scores_communication",  # will be dropped
        "review_scores_location",  # will be dropped
        "review_scores_value",  # will be dropped
        "instant_bookable",  # to encode t==1, f==0
        "calculated_host_listings_count",
        # "calculated_host_listings_count_entire_homes",
        # "calculated_host_listings_count_private_rooms",
        # "calculated_host_listings_count_shared_rooms",
        "reviews_per_month",
    ],
}

ONE_HOT_ENCODE = ["neighbourhood_group_cleansed", "property_type", "room_type"]

MULTI_HOT_ENCODE = "amenities"

BOOL_ENCODE = ["available", "instant_bookable"]

REVIEW_SCORES = [
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]

FILLNA_ZERO = [
    "availability_30",
    "availability_60",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "calculated_host_listings_count",
    "reviews_per_month",
]
