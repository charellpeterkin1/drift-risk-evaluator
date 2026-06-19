def calculate_drift_score(start_score, end_score):
    return end_score - start_score


print("Examples")

print("No Drift:", calculate_drift_score(0, 0))
print("Slight Drift:", calculate_drift_score(0, 1))
print("Major Drift:", calculate_drift_score(0, 3))
print("Improved Safety:", calculate_drift_score(2, 0))
