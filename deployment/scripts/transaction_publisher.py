"""
Transaction Publisher - Simulates Real-Time Credit Card Transactions
Reads creditcard.csv and publishes transactions to Google Cloud Pub/Sub
to simulate a live transaction stream for fraud detection testing.

Usage:
    python transaction_publisher.py --project-id YOUR_PROJECT_ID --rate 10

Arguments:
    --project-id: Your GCP project ID
    --topic-id: Pub/Sub topic name (default: transactions)
    --rate: Transactions per second (default: 10)
    --dataset: Path to creditcard.csv file
    --limit: Maximum number of transactions to send (optional)
    --fraud-only: Only send fraudulent transactions (for testing)
"""

import json
import time
import argparse
import pandas as pd
from google.cloud import pubsub_v1
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Publish credit card transactions to Pub/Sub'
    )
    parser.add_argument(
        '--project-id',
        required=True,
        help='GCP Project ID'
    )
    parser.add_argument(
        '--topic-id',
        default='transactions',
        help='Pub/Sub topic name (default: transactions)'
    )
    parser.add_argument(
        '--dataset',
        default='Fraud-Detection/creditcard.csv',
        help='Path to creditcard.csv file'
    )
    parser.add_argument(
        '--rate',
        type=float,
        default=10.0,
        help='Transactions per second (default: 10)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of transactions to send'
    )
    parser.add_argument(
        '--fraud-only',
        action='store_true',
        help='Only send fraudulent transactions (Class=1)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information for each transaction'
    )

    return parser.parse_args()


def load_dataset(file_path, fraud_only=False):
    """Load the creditcard.csv dataset"""
    print(f"Loading dataset from: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        exit(1)

    print(f"Loaded {len(df):,} transactions")

    if fraud_only:
        df = df[df['Class'] == 1]
        print(f"Filtered to {len(df):,} fraudulent transactions only")

    return df


def create_transaction_message(row, idx):
    """
    Convert a DataFrame row into a transaction message.

    Returns a dict with transaction_id and all features (V1-V28, Amount).
    """
    transaction = {
        'transaction_id': f'txn_{idx}_{int(time.time())}',
        'timestamp': datetime.utcnow().isoformat(),
    }

    # Add all V features
    for i in range(1, 29):
        feature_name = f'V{i}'
        transaction[feature_name] = float(row[feature_name])

    # Add Amount
    transaction['Amount'] = float(row['Amount'])

    # Include ground truth for validation (optional)
    transaction['ActualClass'] = int(row['Class'])

    return transaction


def publish_transactions(project_id, topic_id, df, rate, limit=None, verbose=False):
    """
    Publish transactions to Pub/Sub at specified rate.

    Args:
        project_id: GCP project ID
        topic_id: Pub/Sub topic name
        df: DataFrame with transactions
        rate: Transactions per second
        limit: Maximum number of transactions (None = all)
        verbose: Print details for each transaction
    """
    # Initialize Pub/Sub publisher
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    print("\nStarting transaction stream...")
    print(f"Topic: {topic_path}")
    print(f"Rate: {rate} transactions/second")
    print(f"Total to send: {limit if limit else len(df):,}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Calculate sleep time between messages
    sleep_time = 1.0 / rate if rate > 0 else 0

    # Track statistics
    stats = {
        'total_sent': 0,
        'fraud_sent': 0,
        'normal_sent': 0,
        'errors': 0,
        'start_time': time.time()
    }

    try:
        for idx, row in df.iterrows():
            # Check limit
            if limit and stats['total_sent'] >= limit:
                break

            # Create transaction message
            transaction = create_transaction_message(row, idx)
            is_fraud = transaction['ActualClass'] == 1

            # Publish to Pub/Sub
            try:
                message_json = json.dumps(transaction)
                future = publisher.publish(
                    topic_path,
                    message_json.encode('utf-8')
                )
                message_id = future.result(timeout=5.0)  # Wait for confirmation

                # Update statistics
                stats['total_sent'] += 1
                if is_fraud:
                    stats['fraud_sent'] += 1
                else:
                    stats['normal_sent'] += 1

                # Print progress
                if verbose:
                    fraud_indicator = "FRAUD" if is_fraud else "NORMAL"
                    print(f"[{stats['total_sent']:5d}] {fraud_indicator} | "
                          f"ID: {transaction['transaction_id']} | "
                          f"Amount: ${transaction['Amount']:8.2f} | "
                          f"Msg ID: {message_id}")
                elif stats['total_sent'] % 100 == 0:
                    # Print summary every 100 transactions
                    elapsed = time.time() - stats['start_time']
                    actual_rate = stats['total_sent'] / elapsed if elapsed > 0 else 0
                    print(f"Progress: {stats['total_sent']:,} sent | "
                          f"Rate: {actual_rate:.1f}/sec | "
                          f"Fraud: {stats['fraud_sent']}")

            except Exception as e:
                stats['errors'] += 1
                print(f"ERROR: Failed to publish transaction {idx}: {e}")

            # Rate limiting
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Print final statistics
    elapsed_time = time.time() - stats['start_time']
    actual_rate = stats['total_sent'] / elapsed_time if elapsed_time > 0 else 0

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Total transactions sent: {stats['total_sent']:,}")
    print(f"  Normal: {stats['normal_sent']:,}")
    print(f"  Fraudulent: {stats['fraud_sent']:,}")
    print(f"Errors: {stats['errors']}")
    print(f"Elapsed time: {elapsed_time:.1f} seconds")
    print(f"Actual rate: {actual_rate:.2f} transactions/second")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return stats


def main():
    """Main entry point."""
    args = parse_args()

    # Load dataset
    df = load_dataset(args.dataset, fraud_only=args.fraud_only)

    # Publish transactions
    stats = publish_transactions(
        project_id=args.project_id,
        topic_id=args.topic_id,
        df=df,
        rate=args.rate,
        limit=args.limit,
        verbose=args.verbose
    )

    print("\nTransaction publishing complete!")

    if stats['fraud_sent'] > 0:
        print(f"\nCheck Cloud Function logs to see fraud alerts for "
              f"{stats['fraud_sent']} fraudulent transactions")

    print(f"\nQuery BigQuery to analyze results:")
    print(f"   SELECT * FROM `{args.project_id}.fraud_dataset.predictions` "
          f"ORDER BY timestamp DESC LIMIT 100;")


if __name__ == "__main__":
    main()
