#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Dict, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.types import ArrayType, IntegerType

def calculate_ndcg(predictions: List[int], ground_truth: List[int], k: int = 100) -> float:
    """Calculate NDCG@k for a single user."""
    if not ground_truth:
        return 0.0
    
    # Create ideal ranking (ground truth in order)
    ideal_ranking = ground_truth[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(predictions[:k]):
        if item in ground_truth:
            dcg += 1.0 / (F.log2(i + 2))  # log2(i+2) because i starts at 0
    
    # Calculate IDCG
    idcg = 0.0
    for i, item in enumerate(ideal_ranking):
        idcg += 1.0 / (F.log2(i + 2))
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_map(predictions: List[int], ground_truth: List[int], k: int = 100) -> float:
    """Calculate MAP@k for a single user."""
    if not ground_truth:
        return 0.0
    
    # Calculate precision at each position
    precisions = []
    correct = 0
    
    for i, item in enumerate(predictions[:k]):
        if item in ground_truth:
            correct += 1
            precisions.append(correct / (i + 1))
    
    return sum(precisions) / len(ground_truth) if ground_truth else 0.0

def evaluate_model(
    spark: SparkSession,
    interactions_test: DataFrame,
    recommendations: DataFrame,
    interactions_train: DataFrame,
    tracks_train: DataFrame,
    model_type: str = "popularity",
    k_values: List[int] = [5, 10, 20, 50, 100]
) -> Dict[str, Dict[str, float]]:
    """Evaluate the recommendation model using multiple metrics."""
    print(f"Evaluating {model_type} model...")
    
    # Join test interactions with track data
    interactions_test = interactions_test.join(
        tracks_train,
        on='recording_msid',
        how='left'
    ).select("user_id", "track_id", "track_string_id")
    
    print(f"Test interactions after join: {interactions_test.count()}")
    
    # Create ground truth
    ground_truth = interactions_test.groupby("user_id").agg(
        F.collect_list("track_id").alias("ground_truth")
    )
    
    # Prepare recommendations
    if model_type == "popularity":
        # For popularity model, recommendations are the same for all users
        recomm = recommendations.withColumn("dummy", F.lit(1))
        recomm = recomm.groupby("dummy").agg(
            F.collect_list("track_id").alias("recommendations")
        )
        
        # Join with ground truth
        evaluation_data = ground_truth.withColumn("dummy", F.lit(1))
        evaluation_data = evaluation_data.join(
            recomm,
            on="dummy",
            how="left"
        ).select("user_id", "ground_truth", "recommendations")
    else:
        # For other models, recommendations are user-specific
        evaluation_data = ground_truth.join(
            recommendations,
            on="user_id",
            how="left"
        ).select("user_id", "ground_truth", "recommendations")
    
    # Convert to RDD for RankingMetrics
    evaluation_rdd = evaluation_data.select("recommendations", "ground_truth").rdd
    
    # Calculate metrics for each k value
    results = {}
    for k in k_values:
        metrics = RankingMetrics(evaluation_rdd)
        results[k] = {
            "ndcg": metrics.ndcgAt(k),
            "map": metrics.meanAveragePrecisionAt(k)
        }
        print(f"Metrics at k={k}:")
        print(f"  NDCG: {results[k]['ndcg']:.4f}")
        print(f"  MAP: {results[k]['map']:.4f}")
    
    return results

def evaluate_cold_start(
    spark: SparkSession,
    interactions_test: DataFrame,
    recommendations: DataFrame,
    interactions_train: DataFrame,
    k: int = 100
) -> Dict[str, float]:
    """Evaluate model performance on cold-start users."""
    # Get users in test set but not in train set
    train_users = interactions_train.select("user_id").distinct()
    test_users = interactions_test.select("user_id").distinct()
    cold_start_users = test_users.join(
        train_users,
        on="user_id",
        how="left_anti"
    )
    
    if cold_start_users.count() == 0:
        print("No cold-start users found")
        return {}
    
    # Filter test interactions for cold-start users
    cold_start_interactions = interactions_test.join(
        cold_start_users,
        on="user_id",
        how="inner"
    )
    
    # Evaluate on cold-start users
    results = evaluate_model(
        spark,
        cold_start_interactions,
        recommendations,
        interactions_train,
        k_values=[k]
    )
    
    print(f"Cold-start evaluation (k={k}):")
    print(f"  NDCG: {results[k]['ndcg']:.4f}")
    print(f"  MAP: {results[k]['map']:.4f}")
    
    return results[k] 