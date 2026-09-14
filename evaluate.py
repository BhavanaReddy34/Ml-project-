import argparse
from pipeline import test_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    print(f"[INFO] Running evaluation for {args.dataset}")

    _, _, mae, rmse, _, _ = test_model(args.dataset)

    print(f"[RESULT] MAE  = {mae:.4f}")
    print(f"[RESULT] RMSE = {rmse:.4f}")

if __name__ == "__main__":
    main()