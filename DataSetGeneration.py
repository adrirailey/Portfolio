
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible synthetic dataset
rng = np.random.default_rng(42)

N = 250_000
START = pd.Timestamp("2026-09-01")
END = START + pd.Timedelta(days=28)
OUT = Path("contact_center_data")
OUT.mkdir(exist_ok=True)

# Generate realistic ticket arrival timestamps.
# Weekdays and business hours receive more tickets.
accepted = []

while sum(len(x) for x in accepted) < N:
    seconds = rng.integers(
        0, int((END - START).total_seconds()), size=200_000
    )
    dates = START + pd.to_timedelta(seconds, unit="s")

    hours = dates.hour
    weekdays = dates.dayofweek < 5

    hourly_weight = np.select(
        [
            (hours >= 7) & (hours < 10),
            (hours >= 10) & (hours < 13),
            (hours >= 13) & (hours < 17),
            (hours >= 17) & (hours < 21),
            (hours >= 21) & (hours < 23),
        ],
        [1.5, 2.1, 2.3, 1.4, 0.65],
        default=0.22,
    )

    probability = hourly_weight * np.where(weekdays, 1.0, 0.48) / 2.3
    accepted.append(seconds[rng.random(len(seconds)) < probability])

seconds = np.concatenate(accepted)[:N]
received = (START + pd.to_timedelta(seconds, unit="s")).sort_values()
received = pd.DatetimeIndex(received)

# Ticket attributes
def choose(values, probabilities):
    return rng.choice(values, size=N, p=probabilities)

channels = choose(
    ["Phone", "Email", "Live Chat", "Web Form", "Social Media"],
    [0.46, 0.18, 0.20, 0.12, 0.04],
)

categories = choose(
    ["Billing", "Technical Support", "Account Access",
     "Order Status", "Cancellation", "Complaints",
     "Product Information", "Returns"],
    [0.18, 0.20, 0.12, 0.16, 0.08, 0.09, 0.10, 0.07],
)

priority = choose(
    ["Low", "Medium", "High", "Urgent"],
    [0.22, 0.48, 0.24, 0.06],
)

teams = choose(
    ["Tier 1", "Tier 2", "Billing", "Retention", "Technical"],
    [0.35, 0.18, 0.17, 0.10, 0.20],
)

# Response and resolution times, in minutes
response_baseline = np.select(
    [
        channels == "Phone",
        channels == "Live Chat",
        channels == "Email",
        channels == "Web Form",
    ],
    [0.2, 0.5, 35, 12],
    default=12,
)

first_response = np.maximum(
    0.1,
    rng.lognormal(np.log(response_baseline), 0.7),
)

category_baseline = np.select(
    [
        categories == "Technical Support",
        categories == "Complaints",
        categories == "Billing",
        categories == "Account Access",
        categories == "Returns",
        categories == "Cancellation",
    ],
    [95, 80, 45, 35, 55, 40],
    default=30,
)

priority_factor = np.select(
    [
        priority == "Urgent",
        priority == "High",
        priority == "Low",
    ],
    [0.55, 0.75, 1.25],
    default=1.0,
)

channel_factor = np.select(
    [
        channels == "Phone",
        channels == "Live Chat",
        channels == "Email",
        channels == "Web Form",
    ],
    [0.55, 0.65, 1.7, 1.4],
    default=1.25,
)

resolution_minutes = np.maximum(
    2,
    rng.lognormal(
        np.log(category_baseline * priority_factor * channel_factor),
        0.65,
    ),
)

resolved = rng.random(N) < 0.91
resolved_at = pd.Series(
    received + pd.to_timedelta(resolution_minutes, unit="m")
)
resolved_at.loc[~resolved] = pd.NaT

status = np.where(
    resolved,
    "Resolved",
    choose(["Open", "Pending", "Escalated"], [0.42, 0.38, 0.20]),
)

sla_target = np.select(
    [
        priority == "Urgent",
        priority == "High",
        priority == "Medium",
        priority == "Low",
    ],
    [5, 15, 60, 240],
)

csat = rng.choice([1, 2, 3, 4, 5], N,
                  p=[0.05, 0.08, 0.18, 0.34, 0.35]).astype(float)
csat[rng.random(N) < 0.36] = np.nan

# Assemble the dataset
df = pd.DataFrame({
    "ticket_id": [f"T{i:09d}" for i in range(1, N + 1)],
    "received_at": received,
    "received_date": received.strftime("%Y-%m-%d"),
    "received_hour": received.hour,
    "day_of_week": received.day_name(),
    "channel": channels,
    "ticket_type": categories,
    "priority": priority,
    "customer_type": choose(
        ["New", "Existing", "VIP"], [0.16, 0.78, 0.06]
    ),
    "language": choose(
        ["English", "Spanish", "Portuguese"], [0.38, 0.57, 0.05]
    ),
    "region": choose(
        ["Costa Rica", "Mexico", "Colombia", "United States", "Other"],
        [0.30, 0.22, 0.18, 0.20, 0.10],
    ),
    "assigned_team": teams,
    "agent_id": [f"A{i:04d}" for i in rng.integers(1, 701, N)],
    "customer_id": [f"C{i:07d}" for i in rng.integers(1, 120001, N)],
    "status": status,
    "first_response_minutes": np.round(first_response, 2),
    "resolution_minutes": np.round(resolution_minutes, 2),
    "resolved_at": resolved_at,
    "csat_score": csat,
    "sla_target_minutes": sla_target,
    "sla_met": first_response <= sla_target,
})

# Save both formats
df.to_csv(OUT / "tickets.csv", index=False)

df.to_parquet(
    OUT / "tickets.parquet",
    engine="pyarrow",
    index=False,
    coerce_timestamps="us",
    allow_truncated_timestamps=True
)

print(f"Tickets generated: {len(df):,}")
print(f"First ticket: {df['received_at'].min()}")
print(f"Last ticket:  {df['received_at'].max()}")
print(f"CSV: {OUT / 'tickets.csv'}")
print(f"Parquet: {OUT / 'tickets.parquet'}")


