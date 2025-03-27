#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
from typing import Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count

from ..data.preprocessing import create_track_key, encode_track_string_to_id
from ..evaluation.metrics import evaluate_model

class PopularityRecommender:
    def __init__(self, config: dict):
        """Initialize the popularity-based recommender."""
        self.config = config
        self.recommendations_df = None

    def train(self, spark: SparkSession, interactions_train: DataFrame, tracks_train: DataFrame) -> DataFrame:
        """Train the popularity model and generate recommendations."""
        # Process tracks
        tracks = create_track_key(spark, tracks_train)
        tracks = encode_track_string_to_id(tracks)
        print("SUCCESS: Processed tracks")

        # Join interactions with tracks
        join_df = interactions_train.join(tracks, on="recording_msid", how="left")

        # Calculate track popularity
        count_df = join_df.groupby("track_string_id", "track_id").agg(
            count("*").alias("count")
        )

        # Generate recommendations
        self.recommendations_df = count_df.sort(
            col("count").desc()
        ).select("track_string_id", "track_id").limit(
            self.config['models']['popularity']['n_recommendations']
        )

        return self.recommendations_df

    def predict(self, n_recommendations: int = None) -> DataFrame:
        """Get recommendations from the trained model."""
        if self.recommendations_df is None:
            raise ValueError("Model must be trained before making predictions")
        
        if n_recommendations is None:
            n_recommendations = self.config['models']['popularity']['n_recommendations']
        
        return self.recommendations_df.limit(n_recommendations)

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName('popularity_recommender').getOrCreate()
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize recommender
    recommender = PopularityRecommender(config)
    
    # Load data
    interactions_train = spark.read.parquet(config['data']['raw']['interactions'])
    tracks_train = spark.read.parquet(config['data']['raw']['tracks'])
    interactions_test = spark.read.parquet(config['data']['raw']['interactions_test'])
    
    # Train model and get recommendations
    recommendations = recommender.train(spark, interactions_train, tracks_train)
    
    # Evaluate model
    evaluate_model(
        spark,
        interactions_test,
        recommendations,
        interactions_train,
        tracks_train,
        model_type="popularity"
    )
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main() 