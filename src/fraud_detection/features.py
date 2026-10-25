def add_time_features(df):
    df = df.copy()

    df["hour"] = (df["Time"] // 3600) % 24
    df["day"] = df["Time"] // (3600 * 24)

    return df

def add_velocity_features(df, window=3600):
    df = df.sort_values("Time")

    df["tx_count_last_hour"] = (
        df["Time"]
        .rolling(window=window, min_periods=1)
        .count()
    )

    return df

def build_features(df):
    df = add_time_features(df)
    df = add_velocity_features(df)
    return df