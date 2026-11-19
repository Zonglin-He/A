"""Score the existing sealed predictions with a single row-count field."""

import evaluate180
from train import metrics


def endpoint_metrics(values):
    output = metrics(values)
    # The sealed evaluator adds the record count when it builds the CSV row.
    del output['n']
    return output


if __name__=='__main__':
    evaluate180.metrics = endpoint_metrics
    evaluate180.score()
