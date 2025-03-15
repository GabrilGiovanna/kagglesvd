import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', type=str, default='svd-ae', help='rec-model, support [svd-ae, ease, inf-ae]')
    parser.add_argument('--k', type=int, default=148, help='the rank parameter m')
    parser.add_argument('--dataset', type=str, default='ml-1m', help='available datasets: [gowalla, yelp2018, ml-1m]')
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--grid_search', type=int, default=0)
    parser.add_argument('--group_size', type=int, default=5)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--aggregation', type=str, default='Average')
    parser.add_argument('--individual', type=bool, default=True)
    parser.add_argument('--grouping_method', type=str, default='FCMWithPCC')
    parser.add_argument('--similarity_threshold', type=float, default=0.5)

    return parser.parse_args()