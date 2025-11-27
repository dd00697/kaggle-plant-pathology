from ml.config import CONFIG

def main():
    print("Training with config:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    # TODO: later: import get_dls_for_fold, create_learner, etc.

if __name__ == "__main__":
    main()