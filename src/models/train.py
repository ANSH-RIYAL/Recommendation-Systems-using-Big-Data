#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
from pyspark.sql import SparkSession
from typing import Dict, Any

from .popularity import PopularityRecommender
from .collaborative_filtering import CollaborativeFilteringRecommender
from ..evaluation.metrics import evaluate_model, evaluate_cold_start

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_and_evaluate_models(
    spark: SparkSession,
    config: Dict[str, Any],
    interactions_train: Any,
    interactions_test: Any,
    tracks_train: Any
) -> Dict[str, Dict[str, Any]]:
    """Train and evaluate all recommendation models."""
    results = {}
    
    # Train and evaluate popularity model
    if config['models']['popularity']['enabled']:
        print("\nTraining popularity model...")
        popularity_model = PopularityRecommender(config)
        popularity_recommendations = popularity_model.train(
            spark,
            interactions_train,
            tracks_train
        )
        
        popularity_results = evaluate_model(
            spark,
            interactions_test,
            popularity_recommendations,
            interactions_train,
            tracks_train,
            model_type="popularity",
            k_values=config['evaluation']['k_values']
        )
        
        # Evaluate cold-start performance
        cold_start_results = evaluate_cold_start(
            spark,
            interactions_test,
            popularity_recommendations,
            interactions_train,
            k=config['evaluation']['k_values'][-1]
        )
        
        results['popularity'] = {
            'metrics': popularity_results,
            'cold_start': cold_start_results
        }
    
    # Train and evaluate collaborative filtering model
    if config['models']['collaborative_filtering']['enabled']:
        print("\nTraining collaborative filtering model...")
        cf_model = CollaborativeFilteringRecommender(config)
        cf_model.train(spark, interactions_train, tracks_train)
        
        # Generate recommendations for all users
        all_users = interactions_train.select("user_id").distinct()
        cf_recommendations = cf_model.predict(
            all_users.select("user_id").first()[0],
            n_recommendations=config['models']['collaborative_filtering']['n_recommendations']
        )
        
        cf_results = evaluate_model(
            spark,
            interactions_test,
            cf_recommendations,
            interactions_train,
            tracks_train,
            model_type="collaborative_filtering",
            k_values=config['evaluation']['k_values']
        )
        
        # Evaluate cold-start performance
        cold_start_results = evaluate_cold_start(
            spark,
            interactions_test,
            cf_recommendations,
            interactions_train,
            k=config['evaluation']['k_values'][-1]
        )
        
        results['collaborative_filtering'] = {
            'metrics': cf_results,
            'cold_start': cold_start_results
        }
    
    return results

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName('recommender_training').getOrCreate()
    
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Load data
    interactions_train = spark.read.parquet(config['data']['raw']['interactions'])
    interactions_test = spark.read.parquet(config['data']['raw']['interactions_test'])
    tracks_train = spark.read.parquet(config['data']['raw']['tracks'])
    
    # Train and evaluate models
    results = train_and_evaluate_models(
        spark,
        config,
        interactions_train,
        interactions_test,
        tracks_train
    )
    
    # Print final results
    print("\nFinal Results:")
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()} Model:")
        print("Metrics:")
        for k, metrics in model_results['metrics'].items():
            print(f"  k={k}:")
            print(f"    NDCG: {metrics['ndcg']:.4f}")
            print(f"    MAP: {metrics['map']:.4f}")
        print("Cold-start Performance:")
        print(f"  NDCG: {model_results['cold_start']['ndcg']:.4f}")
        print(f"  MAP: {model_results['cold_start']['map']:.4f}")
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main() 