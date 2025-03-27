#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
from typing import Tuple, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, IndexToString

from ..data.preprocessing import create_track_key, encode_track_string_to_id
from ..evaluation.metrics import evaluate_model

class CollaborativeFilteringRecommender:
    def __init__(self, config: dict):
        """Initialize the collaborative filtering recommender."""
        self.config = config
        self.model = None
        self.user_indexer = None
        self.track_indexer = None
        self.user_converter = None
        self.track_converter = None

    def _create_interaction_matrix(self, interactions: DataFrame, tracks: DataFrame) -> DataFrame:
        """Create the user-item interaction matrix."""
        # Join interactions with tracks
        Xtrain = interactions.join(tracks, on="recording_msid", how="left")
        
        # Select relevant columns
        Xtrain = Xtrain.select("user_id", "track_string_id")
        
        # Encode track IDs
        Xtrain = encode_track_string_to_id(Xtrain)
        
        # Create interaction counts
        Xtrain = Xtrain.groupby("user_id", "track_id").agg(
            F.count("*").alias("count")
        )
        
        return Xtrain

    def _index_users_and_items(self, interactions: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Create and apply indexers for users and items."""
        # Create indexers
        self.user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx")
        self.track_indexer = StringIndexer(inputCol="track_id", outputCol="track_idx")
        
        # Fit and transform
        indexed_data = self.user_indexer.fit(interactions).transform(interactions)
        indexed_data = self.track_indexer.fit(indexed_data).transform(indexed_data)
        
        # Create converters for later use
        self.user_converter = IndexToString(inputCol="user_idx", outputCol="user_id")
        self.track_converter = IndexToString(inputCol="track_idx", outputCol="track_id")
        
        return indexed_data

    def train(self, spark: SparkSession, interactions_train: DataFrame, tracks_train: DataFrame) -> None:
        """Train the collaborative filtering model."""
        # Process tracks
        tracks = create_track_key(spark, tracks_train)
        print("SUCCESS: Processed tracks")

        # Create interaction matrix
        Xtrain = self._create_interaction_matrix(interactions_train, tracks)
        print("SUCCESS: Created interaction matrix")

        # Index users and items
        indexed_data = self._index_users_and_items(Xtrain)
        print("SUCCESS: Indexed users and items")

        # Train ALS model
        self.model = ALS(
            seed=self.config['training']['random_state'],
            implicitPrefs=True,
            rank=self.config['models']['collaborative_filtering']['rank'],
            regParam=self.config['models']['collaborative_filtering']['reg_param'],
            maxIter=self.config['models']['collaborative_filtering']['max_iter'],
            alpha=self.config['models']['collaborative_filtering']['alpha']
        )
        
        self.model.setUserCol("user_idx").setItemCol("track_idx").setRatingCol("count")
        self.model = self.model.fit(indexed_data)
        print("SUCCESS: Trained ALS model")

    def predict(self, user_id: str, n_recommendations: int = None) -> DataFrame:
        """Generate recommendations for a specific user."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if n_recommendations is None:
            n_recommendations = self.config['models']['collaborative_filtering']['n_recommendations']
        
        # Create user DataFrame
        user_df = self.user_indexer.transform(
            self.model.userFactors.select("user_idx").distinct()
        )
        
        # Generate recommendations
        recommendations = self.model.recommendForUserSubset(
            user_df,
            n_recommendations
        )
        
        # Convert indices back to IDs
        recommendations = self.user_converter.transform(recommendations)
        recommendations = self.track_converter.transform(recommendations)
        
        return recommendations

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName('collaborative_filtering').getOrCreate()
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize recommender
    recommender = CollaborativeFilteringRecommender(config)
    
    # Load data
    interactions_train = spark.read.parquet(config['data']['raw']['interactions'])
    tracks_train = spark.read.parquet(config['data']['raw']['tracks'])
    interactions_test = spark.read.parquet(config['data']['raw']['interactions_test'])
    
    # Train model
    recommender.train(spark, interactions_train, tracks_train)
    
    # Generate recommendations for a sample user
    sample_user = interactions_train.select("user_id").first()[0]
    recommendations = recommender.predict(sample_user)
    
    # Evaluate model
    evaluate_model(
        spark,
        interactions_test,
        recommendations,
        interactions_train,
        tracks_train,
        model_type="collaborative_filtering"
    )
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main() 