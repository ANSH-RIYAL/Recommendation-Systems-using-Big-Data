#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Optional

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_track_key(spark: SparkSession, file_path_tracks: str):
    """Create unique track keys from MBID or MSID."""
    tracks = spark.read.parquet(file_path_tracks)
    
    def create_key(mbid_col, msid_col):
        return F.when(F.col("recording_mbid").isNotNull(), mbid_col).otherwise(msid_col)
    
    tracks_new = tracks.withColumn(
        "track_string_id", 
        create_key(F.col("recording_mbid"), F.col("recording_msid"))
    )
    
    return tracks_new

def encode_track_string_to_id(track_df):
    """Encode track string IDs to numeric IDs."""
    unique_keys = track_df.select('track_string_id').distinct().rdd.zipWithIndex().toDF()
    unique_keys = unique_keys.select(F.col("_1.*"), F.col("_2").alias('track_id'))
    track_df = track_df.join(unique_keys, on="track_string_id", how="left")
    return track_df

def preprocess_tracks(spark: SparkSession, config: dict, user_id: str):
    """Main preprocessing function for tracks data."""
    # Load tracks data
    tracks = create_track_key(spark, config['data']['raw']['tracks'])
    print("SUCCESS: Created track keys")

    # Create unique IDs
    unique_keys = tracks.select('key').distinct().withColumn('track_id', F.monotonically_increasing_id())
    print("SUCCESS: Created unique keys")

    # Join with original data
    tracks = tracks.join(unique_keys, on="key", how="left")

    # Save track key mapping
    output_path = os.path.join(config['data']['processed']['output_dir'], 'track_key_mapping.parquet')
    tracks.select("key", "recording_msid", "track_id").write.parquet(
        output_path,
        mode="overwrite"
    )
    print(f"SUCCESS: Saved track key mapping to {output_path}")

    return tracks

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName('track_preprocessing').getOrCreate()
    
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Get user ID
    user_id = os.environ['USER']
    
    # Run preprocessing
    preprocess_tracks(spark, config, user_id)
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main() 