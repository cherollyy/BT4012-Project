"""
Join IEEE-CIS Transaction and Identity datasets using TransactionID as the key.
"""
import pandas as pd

# Load datasets
transaction = pd.read_csv(r'C:\Users\chery\Documents\Documents\uni\y3s1\BT4012\project\datasetjoining\test_transaction.csv')
identity = pd.read_csv(r'C:\Users\chery\Documents\Documents\uni\y3s1\BT4012\project\datasetjoining\test_identity.csv')

print(f"Transaction records: {len(transaction):,}")
print(f"Identity records: {len(identity):,}")

# === JOIN ON COMMON COLUMN: TransactionID ===
# Use LEFT JOIN - keeps all transactions, adds identity info where available
merged = pd.merge(
    transaction, 
    identity, 
    on='TransactionID',  # key
    how='left'           # Keep all transactions
)

print(f"Merged records: {len(merged):,}")
print(f"Columns: {len(merged.columns)}")

# Check how many transactions have identity info
has_identity = merged['TransactionID'].isin(identity['TransactionID']).sum()
print(f"Transactions with identity data: {has_identity:,} ({has_identity/len(merged)*100:.1f}%)")

# Save merged dataset
merged.to_csv('ieee_merged_test.csv', index=False)
print("\n✅ Saved to: ieee_merged_test.csv")

# Display sample
print("\nSample of merged data:")
print(merged.head())