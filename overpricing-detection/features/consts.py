DATA_TAGS = ["calendar", "listings", "reviews"]

# TODO: check in notebook data analysis which columns are e.g. more than half empty or all empty (like 'calendar updated') or have const value (like has_availability) and drop them from features on which models will be trained

SELECTED_FEATURES = {
    "calendar": ["listing_id", "date", "available", "price"],
    "listings": [
        "id",
        "price",  # strip price from $100.00 format
        "name",
        "description",
        # "neighborhood_overview", # all listings from Berlin, Germany
        "neighbourhood_cleansed",  # districts of Berlin - can filter price in neighbourhoos by that; to one-hot
        "neighbourhood_group_cleansed",  # precise location - e.g. street in district; to one-hot
        "property_type",  # to one-hot
        "room_type",  # to one-hot
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "amenities",  # to one-hot
        "minimum_nights",
        "maximum_nights",
        "minimum_minimum_nights",  # to use if "minimum_nights" are NaN
        # "maximum_minimum_nights", # to use if "maximum_nights" are NaN
        # "minimum_maximum_nights",
        "maximum_maximum_nights",
        # "minimum_nights_avg_ntm",
        # "maximum_nights_avg_ntm",
        "availability_30",
        "availability_60",
        "availability_90",
        "availability_365",
        # "number_of_reviews",
        "number_of_reviews_ltm",
        "number_of_reviews_l30d",
        # "first_review",
        # "last_review",
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
        "instant_bookable",
        "calculated_host_listings_count",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
        "reviews_per_month",
        ###
        # "host_id",
        # "host_since",
        # "host_location",
        # "host_about",
        # "host_response_time",
        # "host_response_rate",
        # "host_acceptance_rate",
        # "host_is_superhost",
        # "host_neighbourhood",
        # "host_listings_count",
        # "host_total_listings_count",
        # "host_identity_verifiedneighbourhood",
    ],
    "reviews": ["listing_id", "date", "comments", "numerical_review"],
}
