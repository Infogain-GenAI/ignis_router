"""
Train all ML router models (KNN, SVM, Graph, MF) using llmrouter-lib.

This script trains each router using the example data from llmrouter-lib
and saves the trained .pkl/.pt files into our models/ directory.

Usage:
    cd D:\llm_router_accelerator
    python scripts/train_all_routers.py

Prerequisites:
    - llmrouter-lib installed (pip install -e path/to/LLMRouter)
    - Example data available in llmrouter-lib's data/ folder
"""

import os
import shutil
import sys
from pathlib import Path


def get_llmrouter_root() -> str:
    """Get the root directory of the installed llmrouter-lib package."""
    import llmrouter
    return str(Path(llmrouter.__file__).resolve().parent.parent)


def get_project_root() -> str:
    """Get this project's root directory."""
    return str(Path(__file__).resolve().parents[1])


def train_knn():
    """Train KNN router and save model."""
    print("\n" + "=" * 50)
    print("Training KNN Router...")
    print("=" * 50)

    llmrouter_root = get_llmrouter_root()
    project_root = get_project_root()
    yaml_path = os.path.join(project_root, "configs", "ml_routers", "knnrouter.yaml")

    os.chdir(llmrouter_root)

    from llmrouter.models.knnrouter.router import KNNRouter
    from llmrouter.models.knnrouter.trainer import KNNRouterTrainer

    router = KNNRouter(yaml_path=yaml_path)
    trainer = KNNRouterTrainer(router=router)
    trainer.train()

    # Copy trained model to our project
    src = os.path.join(llmrouter_root, "models", "knnrouter", "knnrouter.pkl")
    dst_dir = os.path.join(project_root, "models", "knnrouter")
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, "knnrouter.pkl"))
        print(f"✅ KNN model saved to: {dst_dir}/knnrouter.pkl")
    else:
        # Check llmrouter package internal path
        src_alt = os.path.join(llmrouter_root, "llmrouter", "models", "knnrouter", "knnrouter.pkl")
        if os.path.exists(src_alt):
            shutil.copy2(src_alt, os.path.join(dst_dir, "knnrouter.pkl"))
            print(f"✅ KNN model saved to: {dst_dir}/knnrouter.pkl")
        else:
            print("❌ KNN model file not found after training!")


def train_svm():
    """Train SVM router and save model."""
    print("\n" + "=" * 50)
    print("Training SVM Router...")
    print("=" * 50)

    llmrouter_root = get_llmrouter_root()
    project_root = get_project_root()
    yaml_path = os.path.join(project_root, "configs", "ml_routers", "svmrouter.yaml")

    os.chdir(llmrouter_root)

    from llmrouter.models.svmrouter.router import SVMRouter
    from llmrouter.models.svmrouter.trainer import SVMRouterTrainer

    router = SVMRouter(yaml_path=yaml_path)
    trainer = SVMRouterTrainer(router=router)
    trainer.train()

    # Copy trained model to our project
    src = os.path.join(llmrouter_root, "models", "svmrouter", "svmrouter.pkl")
    dst_dir = os.path.join(project_root, "models", "svmrouter")
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, "svmrouter.pkl"))
        print(f"✅ SVM model saved to: {dst_dir}/svmrouter.pkl")
    else:
        src_alt = os.path.join(llmrouter_root, "llmrouter", "models", "svmrouter", "svmrouter.pkl")
        if os.path.exists(src_alt):
            shutil.copy2(src_alt, os.path.join(dst_dir, "svmrouter.pkl"))
            print(f"✅ SVM model saved to: {dst_dir}/svmrouter.pkl")
        else:
            print("❌ SVM model file not found after training!")


def train_graph():
    """Train Graph router and save model."""
    print("\n" + "=" * 50)
    print("Training Graph Router...")
    print("=" * 50)

    llmrouter_root = get_llmrouter_root()
    project_root = get_project_root()
    yaml_path = os.path.join(project_root, "configs", "ml_routers", "graphrouter.yaml")

    os.chdir(llmrouter_root)

    from llmrouter.models.graphrouter.router import GraphRouter
    from llmrouter.models.graphrouter.trainer import GraphTrainer

    router = GraphRouter(yaml_path=yaml_path)
    trainer = GraphTrainer(router=router)
    trainer.train()

    # Graph router saves .pt file
    src = os.path.join(llmrouter_root, "models", "graphrouter", "graphrouter.pt")
    dst_dir = os.path.join(project_root, "models", "graphrouter")
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, "graphrouter.pt"))
        print(f"✅ Graph model saved to: {dst_dir}/graphrouter.pt")
    else:
        src_alt = os.path.join(llmrouter_root, "llmrouter", "models", "graphrouter", "graphrouter.pt")
        if os.path.exists(src_alt):
            shutil.copy2(src_alt, os.path.join(dst_dir, "graphrouter.pt"))
            print(f"✅ Graph model saved to: {dst_dir}/graphrouter.pt")
        else:
            print("❌ Graph model file not found after training!")


def train_mf():
    """Train MF (Matrix Factorization) router and save model."""
    print("\n" + "=" * 50)
    print("Training MF Router...")
    print("=" * 50)

    llmrouter_root = get_llmrouter_root()
    project_root = get_project_root()
    yaml_path = os.path.join(project_root, "configs", "ml_routers", "mfrouter.yaml")

    os.chdir(llmrouter_root)

    from llmrouter.models.mfrouter.router import MFRouter
    from llmrouter.models.mfrouter.trainer import MFRouterTrainer

    router = MFRouter(yaml_path=yaml_path)
    trainer = MFRouterTrainer(router=router)
    trainer.train()

    # MF router saves .pt file
    src = os.path.join(llmrouter_root, "models", "mfrouter", "mfrouter.pt")
    dst_dir = os.path.join(project_root, "models", "mfrouter")
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, "mfrouter.pt"))
        print(f"✅ MF model saved to: {dst_dir}/mfrouter.pt")
    else:
        src_alt = os.path.join(llmrouter_root, "llmrouter", "models", "mfrouter", "mfrouter.pt")
        if os.path.exists(src_alt):
            shutil.copy2(src_alt, os.path.join(dst_dir, "mfrouter.pt"))
            print(f"✅ MF model saved to: {dst_dir}/mfrouter.pt")
        else:
            print("❌ MF model file not found after training!")


def main():
    print("=" * 50)
    print("LLM Router Accelerator — Train All ML Routers")
    print("=" * 50)
    print(f"Project root: {get_project_root()}")
    print(f"LLMRouter root: {get_llmrouter_root()}")

    routers_to_train = sys.argv[1:] if len(sys.argv) > 1 else ["knn", "svm", "graph", "mf"]

    for router_name in routers_to_train:
        try:
            if router_name == "knn":
                train_knn()
            elif router_name == "svm":
                train_svm()
            elif router_name == "graph":
                train_graph()
            elif router_name == "mf":
                train_mf()
            else:
                print(f"❌ Unknown router: {router_name}")
        except Exception as exc:
            print(f"❌ Failed to train {router_name}: {exc}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
