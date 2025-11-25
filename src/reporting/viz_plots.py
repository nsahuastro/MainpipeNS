import matplotlib.pyplot as plt

def plot_summary_percentage(summary_pct): #sample_size=25000
    labels = list(summary_pct.keys())
    values = list(summary_pct.values()) ##[summary[k] / sample_size * 100 for k in labels]

    fig=plt.figure(figsize=(12,6))
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.ylabel("Percentage (%)")
    plt.title("Dataset Quality (percentage)")
    plt.tight_layout()
    return fig

def plot_cleaning_report(counters, title="Cleaning Report"):
    labels = list(counters.keys())
    values = [counters[k] for k in labels]

    fig=plt.figure(figsize=(12,5))
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.ylabel("Document Count")
    plt.title(title)
    plt.tight_layout()
    return fig
