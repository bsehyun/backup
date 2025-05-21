import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_feature_lag_ranges(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    features = df['feature'].unique()
    color_map = plt.get_cmap('tab10')

    for i, feature in enumerate(features):
        sub_df = df[df['feature'] == feature]
        for _, row in sub_df.iterrows():
            ax.barh(
                y=i, 
                width=row['end'] - row['start'], 
                left=row['start'], 
                height=0.4, 
                color=color_map(i), 
                edgecolor='black'
            )
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("Time")
    ax.set_title("Important Lag Periods per Feature")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

# 사용
plot_feature_lag_ranges(merged_df)
