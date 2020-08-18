from metrics import metrics
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

if __name__ == "__main__":
    metrics = Metrics(opts)
    error = metrics.compute_l1_error_on_split()
    print(f'-> L1 Error: {error}')
