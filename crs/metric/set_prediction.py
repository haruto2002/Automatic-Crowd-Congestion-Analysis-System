import argparse
from set_pairwise_pred import set_pairwise_pred
from set_classification_pred import set_classification_pred

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="pairwise_comparison", required=True)
    parser.add_argument("--dataset_type", type=list, default=["pairwise", "classification"], required=True)
    parser.add_argument("--pred_dir", type=str, default="results/0619_debug", required=True)
    parser.add_argument("--dataset_dir", type=str, default="danger_annotation_material/targets", required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    dataset_name = args.dataset_name
    dataset_type = args.dataset_type
    pred_dir = args.pred_dir
    dataset_dir = args.dataset_dir
    if dataset_type == "pairwise":
        set_pairwise_pred(dataset_name, pred_dir, dataset_dir)
    elif dataset_type == "classification":
        set_classification_pred(dataset_name, pred_dir, dataset_dir)


if __name__ == "__main__":
    main()