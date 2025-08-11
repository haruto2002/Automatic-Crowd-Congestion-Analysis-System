import argparse
import os
from calc_precision import main as run_precision
from calc_classification_scores import main as run_classification_scores


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="pairwise_comparison", required=True
    )
    parser.add_argument(
        "--dataset_type", type=list, default=["pairwise", "classification"], required=True
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="danger_annotation_material/targets",
        required=True,
    )
    parser.add_argument(
        "--pred_dir", type=str, default="results/0619_debug", required=True
    )
    parser.add_argument(
        "--eval_column", type=str, default="danger_score_clip", required=True
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    dataset_name = args.dataset_name
    dataset_type = args.dataset_type
    dataset_dir = args.dataset_dir
    pred_dir = args.pred_dir
    eval_column = args.eval_column
    save_dir = f"{pred_dir}/metric_results/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    path2gt = f"{dataset_dir}/GT/merged_gt.csv"
    path2pred = f"{pred_dir}/pred_data/{dataset_name}_pred_data.json"

    print("DATASET NAME:", dataset_name)
    print("DATASET TYPE:", dataset_type)
    print("GT PATH:", path2gt)
    print("PRED PATH:", path2pred)
    print("EVAL COLUMN:", eval_column)
    print("SAVE DIR:", save_dir)
    if dataset_type == "pairwise":
        run_precision(
            save_dir,
            path2gt,
            path2pred,
            eval_column,
        )
    elif dataset_type == "classification":
        run_classification_scores(
            save_dir,
            path2gt,
            path2pred,
            eval_column,
        )


if __name__ == "__main__":
    main()






